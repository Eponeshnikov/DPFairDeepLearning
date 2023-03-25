import torch
from torch import optim
from utils import CMLogger
from progressbar import progressbar
from model import EqualOddModel
from opacus import PrivacyEngine
from sklearn.linear_model import LogisticRegression
from torch.optim.lr_scheduler import PolynomialLR, ConstantLR
from fairness_metrics import fair_scores
import numpy as np
import warnings
import time

warnings.simplefilter("ignore")


def str2optimizer(stropt):
    if stropt == 'RMSprop':
        optimizer = optim.RMSprop
    elif stropt == 'NAdam':
        optimizer = optim.NAdam
    else:
        optimizer = None
        raise Exception('Only RMSprop and NAdam supported')
    return optimizer


def str2scheduler(opt, epoch, scheduler):
    if scheduler[0] == 'PolynomialLR':
        return PolynomialLR(opt, total_iters=epoch, power=scheduler[1])
    elif scheduler[0] == 'ConstantLR':
        return ConstantLR(opt, total_iters=epoch)
    else:
        raise Exception('Only ConstantLR and PolynomialLR supported')


def str2eval_model(streval_model):
    if streval_model == 'LR':
        return LogisticRegression(max_iter=1000)
    else:
        raise Exception('Only LogisticRegression (LR) supported')


class Trainer:
    def __init__(self, model, data, trainer_args, privacy_args):
        """Trainer for adversarial fair representation"""
        self.step = None
        torch.backends.cudnn.benchmark = True
        self.device_name = model.device_name
        self.device = torch.device(self.device_name)
        self.eval_model_name = trainer_args.eval_model
        self.eval_model = str2eval_model(self.eval_model_name)

        self.offline_mode = trainer_args.offline_mode

        self.epoch = trainer_args.epoch
        self.adv_on_batch = trainer_args.adv_on_batch
        self.privacy_args = privacy_args
        self.model = model
        self.seed = trainer_args.seed
        self.clip_grad = {'ae': trainer_args.grad_clip_ae, 'adv': trainer_args.grad_clip_adv,
                          'class': trainer_args.grad_clip_class}
        self.eval_step_fair = trainer_args.eval_step_fair
        self.epoch_plt = {"encoder": 0, "classifier": 0, "adversary": 0}
        self.params_plt = {}
        # optimizer for encoder-classifier nets
        self.encoder_class_op = str2optimizer(trainer_args.optimizer_enc_class)(
            self.model.encoderclassifier.parameters(), lr=trainer_args.lr_enc_class)
        # optimizer for adversary nets
        self.adversary_op = str2optimizer(trainer_args.optimizer_adv)(
            self.model.adversary.parameters(), lr=trainer_args.lr_adv)
        self.enc_class_sch = str2scheduler(
            self.encoder_class_op, self.epoch, (trainer_args.enc_class_sch, trainer_args.enc_class_sch_pow))
        self.adv_sch = str2scheduler(
            self.adversary_op, self.epoch, (trainer_args.adv_sch, trainer_args.adv_sch_pow))

        self.train_data = data[0]
        self.test_data = data[1]

        self.name = model.name

        self.logger = CMLogger(self.name, trainer_args.dataset, (trainer_args.config_dir, trainer_args.server),
                               trainer_args.offline_mode)
        self.logger.task.add_tags(trainer_args.dataset)
        tags = [self.name, trainer_args.sensattr]
        self.logger.task.add_tags(tags)

        X_test = self.test_data.dataset.X.cpu().detach().numpy()
        S_test = self.test_data.dataset.A.cpu().detach().numpy()
        X_train = self.train_data.dataset.X.cpu().detach().numpy()
        S_train = self.train_data.dataset.A.cpu().detach().numpy()

        dataset_params = {'Train size': len(X_train), 'Test size': len(X_test), 'Sensattr ones train': sum(S_train),
                          'Sensattr ones test': sum(S_test)}
        self.logger.add_params(dataset_params)

    def train_adversary_on_batch(self, batch_data, sensitive_a, label_y):
        """ Train the adversary with fixed classifier-encoder """
        # reset gradient
        self.model.classifier.train()
        self.model.autoencoder.train()
        self.model.adversary.train()
        self.adversary_op.zero_grad()
        self.step += 1

        with torch.no_grad():
            reconst, z = self.model.autoencoder(batch_data)
            # predict class label from latent dimension
            pred_y = self.model.classifier(z)

        adv_input = z

        sentive_feature = sensitive_a

        if isinstance(self.model, EqualOddModel):
            # for equalized odds, the adversary also receives the class label
            adv_input = torch.cat(
                (z, label_y.view(label_y.shape[0], 1)), 1)

        # cl_error = self.model.get_class_loss(pred_y, label_y)
        # rec_error = self.model.get_recon_loss(reconst, batch_data)

        # predict sensitive attribut from latent dimension
        pred_a = self.model.adversary(adv_input)
        # Compute the adversary loss error
        adv_error = self.model.get_adv_loss(pred_a, sentive_feature)

        # Compute the overall loss and take a negative gradient for the adversary
        error = self.model.advweight * adv_error  # -self.model.get_loss(rec_error, cl_error, adv_error, label_y)
        error.backward()
        grad_norms = [self.get_grad_norm(i) for i in ['encoder', 'classifier', 'adversary']]
        self.logger.log_metric("Gradient norms", "Encoder", grad_norms[0], self.step)
        self.logger.log_metric("Gradient norms", "Classifier", grad_norms[1], self.step)
        self.logger.log_metric("Gradient norms", "Adversary", grad_norms[2], self.step)
        if self.clip_grad['adv'] > 0 and 'adversary' not in self.privacy_args.privacy_in:
            torch.nn.utils.clip_grad_norm(self.model.adversary.parameters(), self.clip_grad['adv'])
        self.adversary_op.step()

        return adv_error

    def make_private(self):
        privacy_engines = {"encoder_classifier": PrivacyEngine(),
                           "adversary": PrivacyEngine()}
        private_params = {}
        if self.privacy_args.privacy_in is None:
            self.privacy_args.privacy_in = []
        if len(self.privacy_args.privacy_in) > 0:
            private_params["eps"] = self.privacy_args.eps
            private_params["delta"] = self.privacy_args.delta
            private_params["max_grad_norm"] = self.privacy_args.max_grad_norm
        tags = [i for i in self.privacy_args.privacy_in if i in privacy_engines.keys()]
        if len(tags) > 0:
            tags.append(f"ε={self.privacy_args.eps}")
        self.logger.add_params(private_params)
        self.logger.task.add_tags(tags)
        for part in self.privacy_args.privacy_in:
            if part == 'encoder_classifier':
                gen = torch.Generator(device=self.device_name)
                gen.manual_seed(self.seed)
                self.model.encoderclassifier, self.encoder_class_op, self.train_data = \
                    privacy_engines[part].make_private_with_epsilon(
                        module=self.model.encoderclassifier,
                        optimizer=self.encoder_class_op,
                        data_loader=self.train_data,
                        epochs=self.epoch,
                        target_epsilon=self.privacy_args.eps,
                        target_delta=self.privacy_args.delta,
                        max_grad_norm=self.privacy_args.max_grad_norm,
                        noise_generator=gen
                    )
            elif part == 'adversary':
                gen = torch.Generator(device=self.device_name)
                gen.manual_seed(self.seed + 1)
                self.model.adversary, self.adversary_op, self.train_data = \
                    privacy_engines[part].make_private_with_epsilon(
                        module=self.model.adversary,
                        optimizer=self.adversary_op,
                        data_loader=self.train_data,
                        epochs=self.epoch * self.adv_on_batch,
                        target_epsilon=self.privacy_args.eps,
                        target_delta=self.privacy_args.delta,
                        max_grad_norm=self.privacy_args.max_grad_norm,
                        noise_generator=gen
                    )
        return privacy_engines

    def get_grad_norm(self, model_):
        model = None
        if model_ == 'encoder':
            model = self.model.autoencoder.encoder
        elif model_ == 'classifier':
            model = self.model.classifier
        elif model_ == 'adversary':
            model = self.model.adversary
        total_norm = 0
        parameters = [(pn, p) for pn, p in model.named_parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p[1].grad.detach().data.norm(2)
            # self.logger.log_metric(model_ + " norms", p[0], param_norm, self.epoch_plt[model_])
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.epoch_plt[model_] += 1
        return total_norm

    def train(self):
        """Train with fixed adversary or classifier-encoder-decoder across epoch
        """
        adversary_loss_log = 0
        total_loss_log = 0
        classifier_loss_log = 0
        autoencoder_loss_log = 0
        torch.autograd.set_detect_anomaly(True)
        for n_batch, (train_x, label_y, sensitive_a) in enumerate(self.train_data):
            self.step += 1
            train_data = train_x  # .to(self.device)
            label_y = label_y  # .to(self.device)
            sensitive_a = sensitive_a  # .to(self.device)
            self.model.classifier.train()
            self.model.autoencoder.train()
            self.model.adversary.train()

            # reset the gradients back to zero
            self.encoder_class_op.zero_grad()
            # self.classifier_op.zero_grad()

            # compute reconstruction and latent space  the
            reconstructed, z = self.model.autoencoder(train_data)

            # predict class label from Z
            pred_y = self.model.classifier(z)
            adv_input = z
            # for equalized odds, the adversary also receives the class label
            if isinstance(self.model, EqualOddModel):
                adv_input = torch.cat(
                    (z, label_y.view(label_y.shape[0], 1)), 1)
            # compute the adversary loss
            with torch.no_grad():
                # predict sensitive attribute from Z
                pred_a = self.model.adversary(adv_input)  # fixed adversary
                adversary_loss = self.model.get_adv_loss(pred_a, sensitive_a)
            # compute the classification loss
            classifier_loss = self.model.get_class_loss(pred_y, label_y)
            # compute the reconstruction loss
            autoencoder_loss = self.model.get_recon_loss(reconstructed, train_data)
            # compute the total loss
            total_loss = self.model.get_loss(autoencoder_loss, classifier_loss, adversary_loss, label_y)

            # backpropagate the gradient encoder-classifier with fixed adversary
            total_loss.backward()
            grad_norms = [self.get_grad_norm(i) for i in ['encoder', 'classifier', 'adversary']]
            self.logger.log_metric("Gradient norms", "Encoder", grad_norms[0], self.step)
            self.logger.log_metric("Gradient norms", "Classifier", grad_norms[1], self.step)
            self.logger.log_metric("Gradient norms", "Adversary", grad_norms[2], self.step)
            # update parameter of the classifier and the encoder
            if self.clip_grad['ae'] > 0 and 'encoder_classifier' not in self.privacy_args.privacy_in:
                torch.nn.utils.clip_grad_norm(self.model.autoencoder.encoder.parameters(), self.clip_grad['ae'])
            if self.clip_grad['ae'] > 0 and 'autoencoder' not in self.privacy_args.privacy_in:
                torch.nn.utils.clip_grad_norm(self.model.autoencoder.decoder.parameters(), self.clip_grad['ae'])
            if self.clip_grad['class'] > 0 and 'encoder_classifier' not in self.privacy_args.privacy_in:
                torch.nn.utils.clip_grad_norm(self.model.classifier.parameters(), self.clip_grad['class'])
            # self.classifier_op.step()
            self.encoder_class_op.step()

            adversary_loss = 0
            # train the adversary
            for t in range(self.adv_on_batch):
                # print("update adversary iter=", t)
                adversary_loss += self.train_adversary_on_batch(train_data, sensitive_a, label_y)

            adversary_loss = adversary_loss / self.adv_on_batch

            total_loss_log += total_loss.item()
            classifier_loss_log += classifier_loss.item()
            autoencoder_loss_log += autoencoder_loss.item()
            adversary_loss_log += adversary_loss.item()

        # epoch loss
        self.enc_class_sch.step()
        self.adv_sch.step()
        total_loss_log = total_loss_log / len(self.train_data)
        autoencoder_loss_log = autoencoder_loss_log / len(self.train_data)
        adversary_loss_log = adversary_loss_log / len(self.train_data)
        classifier_loss_log = classifier_loss_log / len(self.train_data)
        return total_loss_log, autoencoder_loss_log, adversary_loss_log, classifier_loss_log

    def test(self):
        adversary_loss_log = 0
        total_loss_log = 0
        classifier_loss_log = 0
        autoencoder_loss_log = 0
        self.model.classifier.eval()
        self.model.autoencoder.eval()
        self.model.adversary.eval()
        with torch.no_grad():
            for n_batch, (test_x, label_y, sensitive_a) in enumerate(self.test_data):
                test_x = test_x  # .to(self.device)
                label_y = label_y  # .to(self.device)
                sensitive_a = sensitive_a  # .to(self.device)
                # compute reconstruction and latent space
                reconstructed, z = self.model.autoencoder(test_x)

                # predict class label from Z
                pred_y = self.model.classifier(z)

                adv_input = z
                if isinstance(self.model, EqualOddModel):
                    adv_input = torch.cat(
                        (z, label_y.view(label_y.shape[0], 1)), 1)
                # predict sensitive attribute from Z
                pred_a = self.model.adversary(adv_input)  # fixed adversary

                # compute the reconstruction loss
                autoencoder_loss = self.model.get_recon_loss(reconstructed, test_x).item()
                # compute the classification loss
                classifier_loss = self.model.get_class_loss(pred_y, label_y).item()
                # compute the adversary loss
                adversary_loss = self.model.get_adv_loss(pred_a, sensitive_a).item()
                # compute the total loss
                total_loss = self.model.get_loss(autoencoder_loss, classifier_loss, adversary_loss, label_y)

                total_loss_log += total_loss
                classifier_loss_log += classifier_loss
                autoencoder_loss_log += autoencoder_loss
                adversary_loss_log += adversary_loss

            total_loss_log = total_loss_log / len(self.train_data)
            autoencoder_loss_log = autoencoder_loss_log / len(self.train_data)
            adversary_loss_log = adversary_loss_log / len(self.train_data)
            classifier_loss_log = classifier_loss_log / len(self.train_data)
            return total_loss_log, autoencoder_loss_log, adversary_loss_log, classifier_loss_log

    def calc_fair_metrics(self):
        results = {}
        clr = self.eval_model
        X_test = self.test_data.dataset.X.cpu().detach().numpy()
        y_test = self.test_data.dataset.y.cpu().detach().numpy()
        S_test = self.test_data.dataset.A.cpu().detach().numpy()
        X_train = self.train_data.dataset.X.cpu().detach().numpy()
        y_train = self.train_data.dataset.y.cpu().detach().numpy()
        S_train = self.train_data.dataset.A.cpu().detach().numpy()
        X_transformed_train = self.model.transform(torch.from_numpy(X_train).to(self.device)).cpu().detach().numpy()
        X_transformed_test = self.model.transform(torch.from_numpy(X_test).to(self.device)).cpu().detach().numpy()

        clr.fit(X_transformed_train, y_train)
        y_pred_test = clr.predict(X_transformed_test)
        y_pred_train = clr.predict(X_transformed_train)
        y_pred_train_nn = torch.round(self.model.classifier(
            self.model.transform(
                torch.from_numpy(X_train).to(self.device)))).cpu().detach().numpy()
        y_pred_test_nn = torch.round(self.model.classifier(
            self.model.transform(
                torch.from_numpy(X_test).to(self.device)))).cpu().detach().numpy()
        acc_, dp_, eqodd_, eopp_ = fair_scores([y_train, y_test, y_pred_train, y_pred_test], [S_train, S_test])
        acc__, dp__, eqodd__, eopp__ = fair_scores([y_train, y_test, y_pred_train_nn, y_pred_test_nn],
                                                   [S_train, S_test])

        results[self.name + ' test'] = (acc_[1], dp_[1], eqodd_[1], eopp_[1],
                                        acc__[1], dp__[1], eqodd__[1], eopp__[1])
        results[self.name + ' train'] = (acc_[0], dp_[0], eqodd_[0], eopp_[0],
                                         acc__[0], dp__[0], eqodd__[0], eopp__[0])
        return results

    def train_process(self):
        self.step = 0
        privacy_engines = self.make_private()
        clr = self.eval_model
        X_test = self.test_data.dataset.X.cpu().detach().numpy()
        y_test = self.test_data.dataset.y.cpu().detach().numpy()
        S_test = self.test_data.dataset.A.cpu().detach().numpy()

        X_train = self.train_data.dataset.X.cpu().detach().numpy()
        y_train = self.train_data.dataset.y.cpu().detach().numpy()
        S_train = self.train_data.dataset.A.cpu().detach().numpy()

        results_ = {}
        clr.fit(X_train, y_train)
        y_pred_test = clr.predict(X_test)
        y_pred_train = clr.predict(X_train)
        acc_, dp_, eqodd_, eopp_ = fair_scores([y_train, y_test, y_pred_train, y_pred_test], [S_train, S_test])
        results_["Unfair test"] = (acc_[1], dp_[1], eqodd_[1], eopp_[1])
        results_["Unfair train"] = (acc_[0], dp_[0], eqodd_[0], eopp_[0])
        results_, results = self.log_metric_test(results_, 0)
        results = {self.name + ' test': [1 for i in range(8)]}
        for epoch in progressbar(range(1, self.epoch + 1)):  # loop over dataset
            # grad_norms = [self.get_grad_norm(i) for i in ['autoencoder', 'classifier', 'adversary']]
            # self.logger.log_metric("Gradient norms", "Autoencoder", grad_norms[0], epoch)
            # self.logger.log_metric("Gradient norms", "Classifier", grad_norms[1], epoch)
            # self.logger.log_metric("Gradient norms", "Adversary", grad_norms[2], epoch)
            # train
            total_loss_train, autoencoder_loss_train, \
                adversary_loss_train, classifier_loss_train = self.train()
            self.logger.log_metric("Autoencoder Loss", "train loss", autoencoder_loss_train, epoch)
            self.logger.log_metric("Adversary Loss", "train loss", adversary_loss_train, epoch)
            self.logger.log_metric("Classifier Loss", "train loss", classifier_loss_train, epoch)
            self.logger.log_metric("Total Loss", "train loss", total_loss_train, epoch)
            if epoch % self.eval_step_fair == 0:
                results_, results = self.log_metric_test(results_, epoch)

            if epoch > 1:
                if 'encoder_classifier' in self.privacy_args.privacy_in:
                    self.logger.log_metric("ε", "encoder_classifier", privacy_engines['encoder_classifier'].get_epsilon(
                        self.privacy_args.delta), epoch)
                if 'adversary' in self.privacy_args.privacy_in:
                    self.logger.log_metric("ε", "adversary", privacy_engines['adversary'].get_epsilon(
                        self.privacy_args.delta), epoch)

        if self.device_name == 'cuda':
            torch.cuda.empty_cache()
        # self.logger.task.close()
        time.sleep(5)
        return results[self.name + ' test'][4], results[self.name + ' test'][5], results[self.name + ' test'][6]

    def log_metric_test(self, results_, epoch):
        total_loss_test, autoencoder_loss_test, \
            adversary_loss_test, classifier_loss_test = self.test()
        self.logger.log_metric("Autoencoder Loss", "test loss", autoencoder_loss_test, epoch)
        self.logger.log_metric("Adversary Loss", "test loss", adversary_loss_test, epoch)
        self.logger.log_metric("Classifier Loss", "test loss", classifier_loss_test, epoch)
        self.logger.log_metric("Total Loss", "test loss", total_loss_test, epoch)
        results = self.calc_fair_metrics()

        self.logger.log_metric("Accuracy", "Unfair test", results_['Unfair test'][0], epoch)
        self.logger.log_metric("Accuracy", "Unfair train", results_['Unfair train'][0], epoch)
        self.logger.log_metric("Accuracy", 'Model' + ' test', results[self.name + ' test'][0], epoch)
        self.logger.log_metric("Accuracy", 'Model' + ' train', results[self.name + ' train'][0], epoch)
        self.logger.log_metric("Accuracy", 'Model' + ' test NN', results[self.name + ' test'][4], epoch)
        self.logger.log_metric("Accuracy", 'Model' + ' train NN', results[self.name + ' train'][4], epoch)

        self.logger.log_metric("ΔDP", "Unfair test", results_['Unfair test'][1], epoch)
        self.logger.log_metric("ΔDP", "Unfair train", results_['Unfair train'][1], epoch)
        self.logger.log_metric("ΔDP", 'Model' + ' test', results[self.name + ' test'][1], epoch)
        self.logger.log_metric("ΔDP", 'Model' + ' train', results[self.name + ' train'][1], epoch)
        self.logger.log_metric("ΔDP", 'Model' + ' test NN', results[self.name + ' test'][5], epoch)
        self.logger.log_metric("ΔDP", 'Model' + ' train NN', results[self.name + ' train'][5], epoch)

        self.logger.log_metric("ΔEOD", "Unfair test", results_['Unfair test'][2], epoch)
        self.logger.log_metric("ΔEOD", "Unfair train", results_['Unfair train'][2], epoch)
        self.logger.log_metric("ΔEOD", 'Model' + ' test', results[self.name + ' test'][2], epoch)
        self.logger.log_metric("ΔEOD", 'Model' + ' train', results[self.name + ' train'][2], epoch)
        self.logger.log_metric("ΔEOD", 'Model' + ' test NN', results[self.name + ' test'][6], epoch)
        self.logger.log_metric("ΔEOD", 'Model' + ' train NN', results[self.name + ' train'][6], epoch)

        self.logger.log_metric("ΔEOP", "Unfair test", results_['Unfair test'][3], epoch)
        self.logger.log_metric("ΔEOP", "Unfair train", results_['Unfair train'][3], epoch)
        self.logger.log_metric("ΔEOP", 'Model' + ' test', results[self.name + ' test'][3], epoch)
        self.logger.log_metric("ΔEOP", 'Model' + ' train', results[self.name + ' train'][3], epoch)
        self.logger.log_metric("ΔEOP", 'Model' + ' test NN', results[self.name + ' test'][7], epoch)
        self.logger.log_metric("ΔEOP", 'Model' + ' train NN', results[self.name + ' train'][7], epoch)

        self.logger.log_metric("Test Acc/Fair", "DP Unfair",
                               results_['Unfair test'][0] / (1 + results_['Unfair test'][1]),
                               epoch)
        self.logger.log_metric("Test Acc/Fair", "EOD Unfair",
                               results_['Unfair test'][0] / (1 + results_['Unfair test'][2]),
                               epoch)
        self.logger.log_metric("Test Acc/Fair", "EOP Unfair",
                               results_['Unfair test'][0] / (1 + results_['Unfair test'][3]),
                               epoch)

        self.logger.log_metric("Test Acc/Fair", "DP",
                               results[self.name + ' test'][0] / (1 + results[self.name + ' test'][1]),
                               epoch)
        self.logger.log_metric("Test Acc/Fair", "EOD",
                               results[self.name + ' test'][0] / (1 + results[self.name + ' test'][2]),
                               epoch)
        self.logger.log_metric("Test Acc/Fair", "EOP",
                               results[self.name + ' test'][0] / (1 + results[self.name + ' test'][3]),
                               epoch)
        self.logger.log_metric("Test Acc/Fair", "DP NN",
                               results[self.name + ' test'][4] / (1 + results[self.name + ' test'][5]),
                               epoch)
        self.logger.log_metric("Test Acc/Fair", "EOD NN",
                               results[self.name + ' test'][4] / (1 + results[self.name + ' test'][6]),
                               epoch)
        self.logger.log_metric("Test Acc/Fair", "EOP NN",
                               results[self.name + ' test'][4] / (1 + results[self.name + ' test'][7]),
                               epoch)
        return results_, results
