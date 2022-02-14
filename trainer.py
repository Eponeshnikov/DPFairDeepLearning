import torch
from torch import nn, optim
from torch.autograd import Variable
from utils import Logger
from progressbar import progressbar
from model import EqualOddModel
from opacus import PrivacyEngine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from helper import plot_results
from fairness_metrics import cross_val_fair_scores
import numpy as np
import warnings
warnings.simplefilter("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, model, data, data_set_name, model_name, privacy_name):
        """Trainer for adversarial fair representation

        Args:
            model (LFRU Model): The model
            data (Dataloader): The dataloader 
            data_set_name (string): Dataset name for logs
            model_name (string): Model's name 
        """
        self.model = model
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
        self.logger = Logger(model_name, data_set_name, privacy_name)

    def save(self):
        self.logger.save_model(self.model.autoencoder, self.name)

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

        # update weights with gradients
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
            self.logger.log(rec_loss_log, clas_loss_log,
                            adv_loss_log, epoch, num_epochs, len(self.train_data))
            # display the epoch training loss
            print("epoch : {}/{}, loss = {:.6f}, adv_loss:{:.6f}, class_loss:{:.6f}, rec_loss:{:.6f}".format(
                epoch + 1, num_epochs, loss_log, adv_loss_log, clas_loss_log, rec_loss_log))

        # self.logger.save_model(self.model.autoencoder, self.name)
        self.logger.close()

    def make_private(self, privacy_modules, privacy_args, num_epochs):
        privacy_engine = PrivacyEngine()
        for part in privacy_modules:
            exec("self.model." + part + ", " + "self." + part + "_op, self.train_data = "
                "privacy_engine.make_private_with_epsilon("
                "module=self.model." + part + ", "
                "optimizer=self." + part + "_op, "
                "data_loader=self.train_data, "
                "epochs=num_epochs, "
                "target_epsilon=privacy_args['" + part + "']['EPSILON'], "
                "target_delta=privacy_args['" + part + "']['DELTA'], "
                "max_grad_norm=privacy_args['" + part + "']['MAX_GRAD_NORM'],)")

    def train(self):
        """Train with fixed adversary or classifier-encoder-decoder across epoch
        """

        adversary_loss_log = 0
        total_loss_log = 0
        classifier_loss_log = 0
        autoencoder_loss_log = 0
        for n_batch, (train_x, label_y, sensitive_a) in enumerate(self.train_data):
            train_data = Variable(train_x)

            train_data = train_data.to(device)
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
            for t in range(10):
                # print("update adversary iter=", t)
                adversary_loss += self.train_adversary_on_batch(train_data, sensitive_a, label_y)

            adversary_loss = adversary_loss / 10

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

                if torch.cuda.is_available():
                    test_data = test_data.to(device)
                    label_y = label_y.cuda()
                    sensitive_a = sensitive_a.cuda()

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

    def calc_fair_metrics(self):
        results = {}
        kfold = KFold(n_splits=5)
        clr = LogisticRegression(max_iter=1000)
        X_test = self.test_data.dataset.X.cpu().detach().numpy()
        y_test = self.test_data.dataset.y.cpu().detach().numpy()
        S_test = self.test_data.dataset.A.cpu().detach().numpy()

        acc_, dp_, eqodd_, eopp_ = cross_val_fair_scores(clr, X_test, y_test, kfold, S_test)
        results["Unfair"] = ([np.mean(acc_), np.mean(dp_), np.mean(eqodd_), np.mean(eopp_)],
                         [np.std(acc_), np.std(dp_), np.std(eqodd_), np.std(eopp_)])

        X_transformed = self.model.transform(torch.from_numpy(X_test).to(device)).cpu().detach().numpy()
        clr = LogisticRegression(max_iter=1000)
        acc_, dp_, eqodd_, eopp_ = cross_val_fair_scores(clr, X_transformed, y_test, kfold, S_test)
        results[self.name] = ([np.mean(acc_), np.mean(dp_), np.mean(eqodd_), np.mean(eopp_)],
                              [np.std(acc_), np.std(dp_), np.std(eqodd_), np.std(eopp_)])
        figs = plot_results(results, show=False)
        return figs

    def train_process(self, privacy_parts, privacy_args, num_epochs=1000):
        self.make_private(privacy_parts, privacy_args, num_epochs)
        for epoch in progressbar(range(1, num_epochs + 1)):  # loop over dataset
            # train
            total_loss_train, autoencoder_loss_train, \
                adversary_loss_train, classifier_loss_train = self.train()
            self.logger.log(autoencoder_loss_train, classifier_loss_train,
                            adversary_loss_train, epoch, num_epochs, len(self.train_data), description='train')
            # test
            total_loss_test, autoencoder_loss_test,\
                adversary_loss_test, classifier_loss_test = self.test()
            self.logger.log(autoencoder_loss_test, adversary_loss_test,
                            classifier_loss_test, epoch, num_epochs, len(self.test_data), description='test')

        figs = self.calc_fair_metrics()
        self.logger.writer.add_figure('Accuracy', figs[0])
        self.logger.writer.add_figure('Fairness metrics', figs[1])
        self.logger.close()


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
