import torch
from torch import optim
from utils import CMLogger
from progressbar import progressbar
from model import EqualOddModel
from opacus import PrivacyEngine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from fairness_metrics import cross_val_fair_scores
import numpy as np
import warnings

warnings.simplefilter("ignore")


def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]


class Trainer:
    def __init__(self, model, data, trainer_args, privacy_args):
        """Trainer for adversarial fair representation"""
        self.device_name = model.device_name
        self.device = torch.device(self.device_name)

        self.epoch = trainer_args.epoch
        self.adv_on_batch = trainer_args.adv_on_batch
        self.privacy_args = privacy_args
        self.model = model
        self.seed = trainer_args.seed
        self.clip_grad = {'ae': trainer_args.grad_clip_ae, 'adv': trainer_args.grad_clip_adv,
                          'class': trainer_args.grad_clip_class}
        self.eval_step_fair = trainer_args.eval_step_fair
        self.epoch_plt = {"autoencoder": 0, "classifier": 0, "adversary": 0}
        self.params_plt = {}
        # optimizer for autoencoder nets
        self.autoencoder_op = optim.RMSprop(self.model.autoencoder.parameters(), lr=0.008)
        # optimizer for classifier nets
        self.classifier_op = optim.RMSprop(
            self.model.classifier.parameters(), lr=0.008)
        # optimizer for adversary nets
        self.adversary_op = optim.RMSprop(self.model.adversary.parameters(), lr=0.008)

        self.train_data = data[0]
        self.test_data = data[1]
        self.name = model.name

        self.logger = CMLogger(self.name, trainer_args.dataset)
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
        """ Train the adversary with fixed classifier-autoencoder """
        # reset gradient
        self.model.classifier.eval()
        self.model.autoencoder.eval()
        self.model.adversary.train()
        self.adversary_op.zero_grad()

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

        cl_error = self.model.get_class_loss(pred_y, label_y)
        rec_error = self.model.get_recon_loss(reconst, batch_data)

        # predict sensitive attribut from latent dimension
        pred_a = self.model.adversary(adv_input)
        # Compute the adversary loss error
        avd_error = self.model.get_adv_loss(pred_a, sentive_feature)

        # Compute the overall loss and take a negative gradient for the adversary
        error = -self.model.get_loss(rec_error, cl_error, avd_error, label_y)
        error.backward()
        if self.clip_grad['adv'] > 0 and 'adversary' not in self.privacy_args.privacy_in:
            torch.nn.utils.clip_grad_norm(self.model.adversary.parameters(), self.clip_grad['adv'])
        self.adversary_op.step()

        return avd_error

    def make_private(self):
        privacy_engines = {"autoencoder": PrivacyEngine(),
                           "adversary": PrivacyEngine(),
                           "classifier": PrivacyEngine()}
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
            if part == 'autoencoder':
                gen = torch.Generator(device=self.device_name)
                gen.manual_seed(self.seed)
                self.model.autoencoder, self.autoencoder_op, self.train_data = \
                    privacy_engines[part].make_private_with_epsilon(
                        module=self.model.autoencoder,
                        optimizer=self.autoencoder_op,
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
            elif part == 'classifier':
                gen = torch.Generator(device=self.device_name)
                gen.manual_seed(self.seed + 2)
                self.model.classifier, self.classifier_op, self.train_data = \
                    privacy_engines[part].make_private_with_epsilon(
                        module=self.model.classifier,
                        optimizer=self.classifier_op,
                        data_loader=self.train_data,
                        epochs=self.epoch,
                        target_epsilon=self.privacy_args.eps,
                        target_delta=self.privacy_args.delta,
                        max_grad_norm=self.privacy_args.max_grad_norm,
                        noise_generator=gen
                    )
        return privacy_engines

    def get_grad_norm(self, model_):
        model = None
        if model_ == 'autoencoder':
            model = self.model.autoencoder
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
            train_data = train_x.to(self.device)
            label_y = label_y.to(self.device)
            sensitive_a = sensitive_a.to(self.device)
            self.model.classifier.train()
            self.model.autoencoder.train()
            self.model.adversary.eval()

            # reset the gradients back to zero
            self.autoencoder_op.zero_grad()
            self.classifier_op.zero_grad()

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

            # backpropagate the gradient encoder-decoder-classifier with fixed adversary
            total_loss.backward()

            # update parameter of the classifier and the autoencoder
            if self.clip_grad['ae'] > 0 and 'autoencoder' not in self.privacy_args.privacy_in:
                torch.nn.utils.clip_grad_norm(self.model.autoencoder.parameters(), self.clip_grad['ae'])
            if self.clip_grad['class'] > 0 and 'classifier' not in self.privacy_args.privacy_in:
                torch.nn.utils.clip_grad_norm(self.model.classifier.parameters(), self.clip_grad['class'])
            self.classifier_op.step()
            self.autoencoder_op.step()

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
                test_x = test_x.to(self.device)
                label_y = label_y.to(self.device)
                sensitive_a = sensitive_a.to(self.device)
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

    def calc_fair_metrics(self, train=False):
        results = {}
        kfold = KFold(n_splits=5)
        clr = LogisticRegression(max_iter=1000)
        X_test = self.test_data.dataset.X.cpu().detach().numpy()
        y_test = self.test_data.dataset.y.cpu().detach().numpy()
        S_test = self.test_data.dataset.A.cpu().detach().numpy()

        X_transformed = self.model.transform(torch.from_numpy(X_test).to(self.device)).cpu().detach().numpy()
        acc_, dp_, eqodd_, eopp_ = cross_val_fair_scores(clr, X_transformed, y_test, kfold, S_test)
        results[self.name + ' test'] = ([np.mean(acc_), np.mean(dp_), np.mean(eqodd_), np.mean(eopp_)],
                                        [np.std(acc_), np.std(dp_), np.std(eqodd_), np.std(eopp_)])
        if train:
            X_train = self.train_data.dataset.X.cpu().detach().numpy()
            y_train = self.train_data.dataset.y.cpu().detach().numpy()
            S_train = self.train_data.dataset.A.cpu().detach().numpy()
            X_transformed = self.model.transform(torch.from_numpy(X_train).to(self.device)).cpu().detach().numpy()
            acc_, dp_, eqodd_, eopp_ = cross_val_fair_scores(clr, X_transformed, y_train, kfold, S_train)
            results[self.name + ' train'] = ([np.mean(acc_), np.mean(dp_), np.mean(eqodd_), np.mean(eopp_)],
                                             [np.std(acc_), np.std(dp_), np.std(eqodd_), np.std(eopp_)])
        # figs = plot_results(results, show=False)
        return results

    def train_process(self):

        privacy_engines = self.make_private()
        kfold = KFold(n_splits=5)
        clr = LogisticRegression(max_iter=1000)
        X_test = self.test_data.dataset.X.cpu().detach().numpy()
        y_test = self.test_data.dataset.y.cpu().detach().numpy()
        S_test = self.test_data.dataset.A.cpu().detach().numpy()

        X_train = self.train_data.dataset.X.cpu().detach().numpy()
        y_train = self.train_data.dataset.y.cpu().detach().numpy()
        S_train = self.train_data.dataset.A.cpu().detach().numpy()

        results_ = {}
        acc_, dp_, eqodd_, eopp_ = cross_val_fair_scores(clr, X_test, y_test, kfold, S_test)
        results_["Unfair test"] = ([np.mean(acc_), np.mean(dp_), np.mean(eqodd_), np.mean(eopp_)],
                                   [np.std(acc_), np.std(dp_), np.std(eqodd_), np.std(eopp_)])
        acc_, dp_, eqodd_, eopp_ = cross_val_fair_scores(clr, X_train, y_train, kfold, S_train)
        results_["Unfair train"] = ([np.mean(acc_), np.mean(dp_), np.mean(eqodd_), np.mean(eopp_)],
                                    [np.std(acc_), np.std(dp_), np.std(eqodd_), np.std(eopp_)])
        for epoch in progressbar(range(1, self.epoch + 1)):  # loop over dataset
            grad_norms = [self.get_grad_norm(i) for i in ['autoencoder', 'classifier', 'adversary']]
            self.logger.log_metric("Gradient norms", "Autoencoder", grad_norms[0], epoch)
            self.logger.log_metric("Gradient norms", "Classifier", grad_norms[1], epoch)
            self.logger.log_metric("Gradient norms", "Adversary", grad_norms[2], epoch)
            # train
            total_loss_train, autoencoder_loss_train, \
            adversary_loss_train, classifier_loss_train = self.train()
            self.logger.log_metric("Autoencoder Loss", "train loss", autoencoder_loss_train, epoch)
            self.logger.log_metric("Adversary Loss", "train loss", adversary_loss_train, epoch)
            self.logger.log_metric("Classifier Loss", "train loss", classifier_loss_train, epoch)

            total_loss_test, autoencoder_loss_test, \
            adversary_loss_test, classifier_loss_test = self.test()
            self.logger.log_metric("Autoencoder Loss", "test loss", autoencoder_loss_test, epoch)
            self.logger.log_metric("Adversary Loss", "test loss", adversary_loss_test, epoch)
            self.logger.log_metric("Classifier Loss", "test loss", classifier_loss_test, epoch)
            if epoch % self.eval_step_fair == 0:
                results = self.calc_fair_metrics(train=True)

                self.logger.log_metric("Accuracy", "Unfair test", results_['Unfair test'][0][0], epoch)
                self.logger.log_metric("Accuracy", "Unfair train", results_['Unfair train'][0][0], epoch)
                self.logger.log_metric("Accuracy", self.name + ' test', results[self.name + ' test'][0][0], epoch)
                self.logger.log_metric("Accuracy", self.name + ' train', results[self.name + ' train'][0][0], epoch)

                self.logger.log_metric("ΔDP", "Unfair test", results_['Unfair test'][0][1], epoch)
                self.logger.log_metric("ΔDP", "Unfair train", results_['Unfair train'][0][1], epoch)
                self.logger.log_metric("ΔDP", self.name + ' test', results[self.name + ' test'][0][1], epoch)
                self.logger.log_metric("ΔDP", self.name + ' train', results[self.name + ' train'][0][1], epoch)

                self.logger.log_metric("ΔEOD", "Unfair test", results_['Unfair test'][0][2], epoch)
                self.logger.log_metric("ΔEOD", "Unfair train", results_['Unfair train'][0][2], epoch)
                self.logger.log_metric("ΔEOD", self.name + ' test', results[self.name + ' test'][0][2], epoch)
                self.logger.log_metric("ΔEOD", self.name + ' train', results[self.name + ' train'][0][2], epoch)

                self.logger.log_metric("ΔEOP", "Unfair test", results_['Unfair test'][0][3], epoch)
                self.logger.log_metric("ΔEOP", "Unfair train", results_['Unfair train'][0][3], epoch)
                self.logger.log_metric("ΔEOP", self.name + ' test', results[self.name + ' test'][0][3], epoch)
                self.logger.log_metric("ΔEOP", self.name + ' train', results[self.name + ' train'][0][3], epoch)

                self.logger.log_metric("Test Acc/Fair", "DP Unfair",
                                       results_['Unfair test'][0][0] / (1 + results_['Unfair test'][0][1]),
                                       epoch)
                self.logger.log_metric("Test Acc/Fair", "EOD Unfair",
                                       results_['Unfair test'][0][0] / (1 + results_['Unfair test'][0][2]),
                                       epoch)
                self.logger.log_metric("Test Acc/Fair", "EOP Unfair",
                                       results_['Unfair test'][0][0] / (1 + results_['Unfair test'][0][3]),
                                       epoch)

                self.logger.log_metric("Test Acc/Fair", "DP",
                                       results[self.name + ' test'][0][0] / (1 + results[self.name + ' test'][0][1]),
                                       epoch)
                self.logger.log_metric("Test Acc/Fair", "EOD",
                                       results[self.name + ' test'][0][0] / (1 + results[self.name + ' test'][0][2]),
                                       epoch)
                self.logger.log_metric("Test Acc/Fair", "EOP",
                                       results[self.name + ' test'][0][0] / (1 + results[self.name + ' test'][0][3]),
                                       epoch)

            self.logger.log_metric("ε", "autoencoder", privacy_engines['autoencoder'].get_epsilon(
                self.privacy_args.delta), epoch)
            self.logger.log_metric("ε", "adversary", privacy_engines['adversary'].get_epsilon(
                self.privacy_args.delta), epoch)
            self.logger.log_metric("ε", "classifier", privacy_engines['classifier'].get_epsilon(
                self.privacy_args.delta), epoch)

        if self.device_name == 'cuda':
            torch.cuda.empty_cache()
