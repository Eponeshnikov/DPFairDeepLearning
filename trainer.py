import torch
from torch import nn, optim
from torch.autograd import Variable
from utils import CMLogger
from progressbar import progressbar
from model import EqualOddModel
from opacus import PrivacyEngine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from fairness_metrics import cross_val_fair_scores
import numpy as np
import warnings
import use_cuda

warnings.simplefilter("ignore")

use_cuda = use_cuda.use_cuda
device_name = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
device = torch.device(device_name)


def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]


class Trainer:
    def __init__(self, model, data, data_set_name, model_name, xavier_weights=False):
        """Trainer for adversarial fair representation

        Args:
            model (LFRU Model): The model
            data (Dataloader): The dataloader 
            data_set_name (string): Dataset name for logs
            model_name (string): Model's name 
        """
        self.model = model
        self.seed = 0
        self.epoch_plt = {"autoencoder":0, "classifier":0, "adversary":0}
        self.params_plt = {}
        # optimizer for autoencoder nets
        self.autoencoder_op = optim.Adam(self.model.autoencoder.parameters(), lr=0.001)
        # optimizer for classifier nets
        self.classifier_op = optim.Adam(
            self.model.classifier.parameters(), lr=0.001)
        # optimizer for adversary nets
        self.adversary_op = optim.Adam(self.model.adversary.parameters(), lr=0.001)

        self.train_data = data[0]
        self.test_data = data[1]
        self.name = model_name  # "{}_{}".format(model_name, model.name)
        model.name = self.name
        # Logger(model_name, model_name)
        # self.logger = Logger(model_name, data_set_name, privacy_name)
        self.logger = CMLogger(model_name, data_set_name)
        self.logger.task.add_tags(data_set_name)
        mainhp = {"Batch size": self.train_data.batch_size,
                  "Autoencoder neurons": self.model.hidden_layers['ae'],
                  "Adversary neurons": self.model.hidden_layers['avd'],
                  "Classifier neurons": self.model.hidden_layers['class'],
                  "Latent dimension": self.model.latent_dim,
                  "Xavier weights init:": str(xavier_weights)}
        tags = [f"batch_size={self.train_data.batch_size}",
                f"neurons={self.model.hidden_layers['ae']}",
                f"latent_dim={self.model.latent_dim}",
                f"xavier_weight={xavier_weights}",
                model_name]
        self.logger.add_params(mainhp)
        self.logger.task.add_tags(tags)
        self.model.autoencoder.float()
        self.model.classifier.float()
        self.model.adversary.float()
        if xavier_weights:
            self.model.autoencoder.apply(self.init_weights)
            self.model.classifier.apply(self.init_weights)
            self.model.adversary.apply(self.init_weights)

    # def save(self):
    #   self.logger.save_model(self.model.autoencoder, self.name)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

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
        # print(error)
        error.backward()

        self.adversary_op.step()

        return avd_error

    def train_autoencoder(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = 0.0
            for n_batch, (train_x, label_y, sensitive_a) in enumerate(self.train_data):
                train_data = Variable(train_x)

                train_data = train_data.to(device)

                self.autoencoder_op.zero_grad()
                # compute reconstruction and latent space  the
                reconstructed, _ = self.model.autoencoder(train_data)

                # compute the recontruction loss
                rec_loss = self.model.get_recon_loss(reconstructed, train_data)

                rec_loss.backward()
                self.autoencoder_op.step()
                train_loss += rec_loss.item() * train_data.size(0)

            # print avg training statistics
            train_loss = train_loss / len(self.train_data)
            print('Epoch: {}/{} \tTraining Loss: {:.6f}'.format(epoch +
                                                                1, num_epochs, train_loss))

    def train2(self, num_epochs=1000):
        """Train with alternate gradient

        Args:
            num_epochs (int, optional): Number of epoch. Defaults to 1000.
        """
        U = False

        for epoch in range(num_epochs):  # loop over dataset
            adv_loss_log = 0
            loss_log = 0
            clas_loss_log = 0
            rec_loss_log = 0
            for n_batch, (train_x, label_y, sensitive_a) in enumerate(self.train_data):
                train_data = Variable(train_x)

                train_data = train_data.to(device)
                label_y = label_y.to(device)
                sensitive_a = sensitive_a.to(device)

                # reset the gradients back to zero
                self.autoencoder_op.zero_grad()
                self.classifier_op.zero_grad()
                self.adversary_op.zero_grad()

                # compute reconstruction and latent space  the
                reconstructed, z = self.model.autoencoder(train_data)

                # predict class label from Z
                pred_y = self.model.classifier(z)
                adv_input = z
                # for equalized odds, the adversary also receives the class label
                if isinstance(self.model, EqualOddModel):
                    adv_input = torch.cat(
                        (z, label_y.view(label_y.shape[0], 1)), 1)

                # predict sentive attribut from Z
                pred_a = self.model.adversary(adv_input)  # fixed adversary
                # compute the adversary loss
                adv_loss = self.model.get_adv_loss(pred_a, sensitive_a)

                # compute the classification loss
                class_loss = self.model.get_class_loss(pred_y, label_y)
                # compute the recontruction loss
                rec_loss = self.model.get_recon_loss(
                    reconstructed, train_data)
                if not U:
                    train_loss = self.model.get_loss(
                        rec_loss, class_loss, adv_loss, label_y)
                    # backpropagate the gradient encoder-decoder-classifier with fixed adversary
                    train_loss.backward()
                    self.classifier_op.step()
                    self.autoencoder_op.step()
                else:
                    train_loss = -self.model.get_loss(
                        rec_loss, class_loss, adv_loss, label_y)
                    # backpropagate the gradient encoder-decoder-classifier with fixed adversary
                    train_loss.backward()
                    self.adversary_op.step()

                U = not U

                # train the adversary
                # for t in range(2):
                # adv_loss = self.train_adversary_on_batch( train_data, sensitive_a, label_y)

                loss_log += train_loss.item()
                clas_loss_log += class_loss.item()
                rec_loss_log += rec_loss.item()
                adv_loss_log += adv_loss.item()
                # print(train_loss)
                # if(n_batch) % 100 == 0:
                # print("epoch : {}/{}, batch = {}, loss = {:.6f}".format(epoch + 1, num_epochs, n_batch, loss_log))
            # epoch loss
            loss_log = loss_log / len(self.train_data)
            rec_loss_log = rec_loss_log / len(self.train_data)
            adv_loss_log = adv_loss_log / len(self.train_data)
            clas_loss_log = clas_loss_log / len(self.train_data)
            # self.logger.log(rec_loss_log, clas_loss_log,
            #                adv_loss_log, epoch, num_epochs, len(self.train_data))
            self.logger.log_metric("Autoencoder Loss", "train loss", rec_loss_log, epoch)
            self.logger.log_metric("Adversary Loss", "train loss", adv_loss_log, epoch)
            self.logger.log_metric("Classifier Loss", "train loss", clas_loss_log, epoch)
            # display the epoch training loss

            print("epoch : {}/{}, loss = {:.6f}, adv_loss:{:.6f}, class_loss:{:.6f}, rec_loss:{:.6f}".format(
                epoch + 1, num_epochs, loss_log, adv_loss_log, clas_loss_log, rec_loss_log))

        # self.logger.save_model(self.model.autoencoder, self.name)

    def make_private(self, privacy_modules, privacy_args, adv_on_batch, num_epochs):
        self.privacy_args = privacy_args
        self.privacy_modules = privacy_modules
        self.num_epochs = num_epochs
        self.adv_on_batch = adv_on_batch
        privacy_engines = {"autoencoder": PrivacyEngine(),
                           "adversary": PrivacyEngine(),
                           "classifier": PrivacyEngine()}
        private_params = {"Private " + i: i in privacy_modules for i in privacy_engines.keys()}
        if len(privacy_modules) > 0:
            private_params["ε"] = privacy_args["EPSILON"]
            private_params["δ"] = privacy_args["DELTA"]
            private_params["MAX_GRAD_NORM"] = privacy_args["MAX_GRAD_NORM"]
        tags = [i for i in privacy_modules if i in privacy_engines.keys()]
        if len(tags) > 0:
            tags.append("ε=" + str(privacy_args["EPSILON"]))
            tags.append("grad_norm=" + str(privacy_args["MAX_GRAD_NORM"]))
        private_params["Adversary on batch"] = int(adv_on_batch)
        tags.append('adv_on_batch=' + str(adv_on_batch))
        self.logger.add_params(private_params)
        self.logger.task.add_tags(tags)
        for part in privacy_modules:
            if part == 'autoencoder':
                gen = torch.Generator(device=device_name)
                gen.manual_seed(self.seed)
                self.model.autoencoder, self.autoencoder_op, self.train_data = \
                    privacy_engines[part].make_private_with_epsilon(
                        module=self.model.autoencoder,
                        optimizer=self.autoencoder_op,
                        data_loader=self.train_data,
                        epochs=num_epochs,
                        target_epsilon=privacy_args['EPSILON'],
                        target_delta=privacy_args['DELTA'],
                        max_grad_norm=privacy_args['MAX_GRAD_NORM'],
                        noise_generator=gen
                    )
            elif part == 'adversary':
                gen = torch.Generator(device=device_name)
                gen.manual_seed(self.seed+1)
                self.model.adversary, self.adversary_op, self.train_data = \
                    privacy_engines[part].make_private_with_epsilon(
                        module=self.model.adversary,
                        optimizer=self.adversary_op,
                        data_loader=self.train_data,
                        epochs=num_epochs * adv_on_batch,
                        target_epsilon=privacy_args['EPSILON'],
                        target_delta=privacy_args['DELTA'],
                        max_grad_norm=privacy_args['MAX_GRAD_NORM'],
                        noise_generator=gen
                    )
            elif part == 'classifier':
                gen = torch.Generator(device=device_name)
                gen.manual_seed(self.seed+2)
                self.model.classifier, self.classifier_op, self.train_data = \
                    privacy_engines[part].make_private_with_epsilon(
                        module=self.model.classifier,
                        optimizer=self.classifier_op,
                        data_loader=self.train_data,
                        epochs=num_epochs,
                        target_epsilon=privacy_args['EPSILON'],
                        target_delta=privacy_args['DELTA'],
                        max_grad_norm=privacy_args['MAX_GRAD_NORM'],
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
            self.logger.log_metric(model_ + " norms", p[0], param_norm, self.epoch_plt[model_])
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.epoch_plt[model_] += 1
        return total_norm

    def train(self, adv_on_batch):
        """Train with fixed adversary or classifier-encoder-decoder across epoch
        """

        adversary_loss_log = 0
        total_loss_log = 0
        classifier_loss_log = 0
        autoencoder_loss_log = 0
        torch.autograd.set_detect_anomaly(True)
        for n_batch, (train_x, label_y, sensitive_a) in enumerate(self.train_data):
            train_data = train_x.to(device)
            label_y = label_y.to(device)
            sensitive_a = sensitive_a.to(device)
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
            self.classifier_op.step()
            self.autoencoder_op.step()

            adversary_loss = 0
            # train the adversary
            for t in range(adv_on_batch):
                # print("update adversary iter=", t)
                adversary_loss += self.train_adversary_on_batch(train_data, sensitive_a, label_y)

            adversary_loss = adversary_loss / adv_on_batch

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
                test_data, label_y, sensitive_a = Variable(
                    test_x), Variable(label_y), Variable(sensitive_a)

                if torch.cuda.is_available() and use_cuda:
                    test_data = test_data.to(device)
                    label_y = label_y.to(device)
                    sensitive_a = sensitive_a.to(device)

                # compute reconstruction and latent space
                reconstructed, z = self.model.autoencoder(test_data)

                # predict class label from Z
                pred_y = self.model.classifier(z)

                adv_input = z
                if isinstance(self.model, EqualOddModel):
                    adv_input = torch.cat(
                        (z, label_y.view(label_y.shape[0], 1)), 1)
                # predict sensitive attribute from Z
                pred_a = self.model.adversary(adv_input)  # fixed adversary

                # compute the reconstruction loss
                autoencoder_loss = self.model.get_recon_loss(reconstructed, test_data).item()
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

        X_transformed = self.model.transform(torch.from_numpy(X_test).to(device)).cpu().detach().numpy()
        acc_, dp_, eqodd_, eopp_ = cross_val_fair_scores(clr, X_transformed, y_test, kfold, S_test)
        results[self.name + ' test'] = ([np.mean(acc_), np.mean(dp_), np.mean(eqodd_), np.mean(eopp_)],
                                        [np.std(acc_), np.std(dp_), np.std(eqodd_), np.std(eopp_)])
        if train:
            X_train = self.train_data.dataset.X.cpu().detach().numpy()
            y_train = self.train_data.dataset.y.cpu().detach().numpy()
            S_train = self.train_data.dataset.A.cpu().detach().numpy()
            X_transformed = self.model.transform(torch.from_numpy(X_train).to(device)).cpu().detach().numpy()
            acc_, dp_, eqodd_, eopp_ = cross_val_fair_scores(clr, X_transformed, y_train, kfold, S_train)
            results[self.name + ' train'] = ([np.mean(acc_), np.mean(dp_), np.mean(eqodd_), np.mean(eopp_)],
                                             [np.std(acc_), np.std(dp_), np.std(eqodd_), np.std(eopp_)])
        # figs = plot_results(results, show=False)
        return results

    def train_process(self, privacy_parts, privacy_args, adv_on_batch, num_epochs=1000):

        privacy_engines = self.make_private(privacy_parts, privacy_args, adv_on_batch, num_epochs)
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
        for epoch in progressbar(range(1, num_epochs + 1)):  # loop over dataset
            grad_norms = [self.get_grad_norm(i) for i in ['autoencoder', 'classifier', 'adversary']]
            self.logger.log_metric("Gradient norms", "Autoencoder", grad_norms[0], epoch)
            self.logger.log_metric("Gradient norms", "Classifier", grad_norms[1], epoch)
            self.logger.log_metric("Gradient norms", "Adversary", grad_norms[2], epoch)
            # train
            total_loss_train, autoencoder_loss_train, \
            adversary_loss_train, classifier_loss_train = self.train(adv_on_batch)
            self.logger.log_metric("Autoencoder Loss", "train loss", autoencoder_loss_train, epoch)
            self.logger.log_metric("Adversary Loss", "train loss", adversary_loss_train, epoch)
            self.logger.log_metric("Classifier Loss", "train loss", classifier_loss_train, epoch)

            total_loss_test, autoencoder_loss_test, \
            adversary_loss_test, classifier_loss_test = self.test()
            self.logger.log_metric("Autoencoder Loss", "test loss", autoencoder_loss_test, epoch)
            self.logger.log_metric("Adversary Loss", "test loss", adversary_loss_test, epoch)
            self.logger.log_metric("Classifier Loss", "test loss", classifier_loss_test, epoch)

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

            self.logger.log_metric("ε", "autoencoder", privacy_engines['autoencoder'].get_epsilon(
                privacy_args['DELTA']), epoch)
            self.logger.log_metric("ε", "adversary", privacy_engines['adversary'].get_epsilon(
                privacy_args['DELTA']), epoch)
            self.logger.log_metric("ε", "classifier", privacy_engines['classifier'].get_epsilon(
                privacy_args['DELTA']), epoch)

        if torch.cuda.is_available() and use_cuda:
            torch.cuda.empty_cache()


'''def train_classifier(classifier, params, is_avd=False):
    """Train a classifier over the dataset with eventually encoded version

      Args:
          classifier (Model): A classifier
          params (array): parameters for the training: number of epoch, dataloader and model name 
          is_avd (bool, optional): [if classifier is trained to predict the senstive attribute]. Defaults to False.
    """
    # loss function
    logger = Logger('classifier', 'data_set_name')
    criteria = nn.BCELoss()

    n_epoch, data, model_name = params
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    for epoch in range(n_epoch):  # loop over dataset
        loss_log = 0
        for n_batch, (train_x, label_y, sentive_a) in enumerate(data):
            train_data = Variable(train_x)
            train_data = train_data.to(device)
            label_y = label_y.to(device).view(-1, 1)
            sentive_a = sentive_a.to(device).view(-1, 1)

            # reset the gradients back to zero
            optimizer.zero_grad()

            prediction = classifier(train_data)

            if is_avd:
                # compute the loss based on sensitive feature
                train_loss = criteria(prediction, sentive_a)
            else:
                # compute the loss based on class label
                train_loss = criteria(prediction, label_y)

            # backpropagate the gradient
            train_loss.backward()

            # update parameter update based on current gradients
            optimizer.step()

            loss_log += train_loss.item()
            loss_log = loss_log / len(data)

            # display the epoch training loss
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, n_epoch, loss_log))
        logger.save_model(classifier, model_name)'''
