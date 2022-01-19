
import torch
from torch import nn, optim
from torch.autograd import Variable
from utils import Logger
from model import EqualOddModel
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer():
    def __init__(self, model, data, data_set_name, model_name):
        """Trainer for adversarial fair representation

        Args:
            model (LFRU Model): The model
            data (Dataloader): The dataloader 
            data_set_name (string): Dataset name for logs
            model_name (string): Model's name 
        """
        self.model = model
        # optimizer for autoencoder nets
        self.gen_op = optim.Adam(self.model.autoencoder.parameters(), lr=0.001)
        # optimizer for classifier nets
        self.class_op = optim.Adam(
            self.model.classifier.parameters(), lr=0.001)
        # optimizer for adversary nets
        self.adver_op = optim.Adam(self.model.adversary.parameters(), lr=0.001)

        self.data = data
        self.name = model_name  # "{}_{}".format(model_name, model.name)
        model.name = self.name
        # Logger(model_name, model_name)
        self.logger = Logger(model_name, data_set_name)

    def save(self):
        self.logger.save_model(self.model.autoencoder, self.name)

    def train_adversary_on_batch(self, batch_data, sentive_a, label_y, epoch=None):
        """ Train the adversary with fixed classifier-autoencoder """
        # reset gradient
        self.adver_op.zero_grad()

        with torch.no_grad():
            reconst, z = self.model.autoencoder(batch_data)

        adv_input = z

        sentive_feature = sentive_a

        if isinstance(self.model, EqualOddModel):
            # for equalized odds, the adversary also receives the class label
            adv_input = torch.cat(
                (z, label_y.view(label_y.shape[0], 1)), 1)

        # predict class label from latent dimension
        pred_y = self.model.classifier(z)

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
        self.adver_op.step()

        if epoch:
            print('Train Adversary Epoch: {}\tLoss: {:.6f}'.format(epoch + 1,  error))

        return error

    def train_autoencoder(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = 0.0
            for n_batch, (train_x, label_y, sensitive_a) in enumerate(self.data):
                train_data = Variable(train_x)

                train_data = train_data.to(device)

                self.gen_op.zero_grad()
                # compute reconstruction and latent space  the
                reconstructed, _ = self.model.autoencoder(train_data)

                # compute the recontruction loss
                rec_loss = self.model.get_recon_loss(reconstructed, train_data)

                rec_loss.backward()
                self.gen_op.step()
                train_loss += rec_loss.item()*train_data.size(0)

            # print avg training statistics
            train_loss = train_loss/len(self.data)
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
            for n_batch, (train_x, label_y, sensitive_a) in enumerate(self.data):
                train_data = Variable(train_x)

                train_data = train_data.to(device)
                label_y = label_y.to(device)
                sensitive_a = sensitive_a.to(device)

                # reset the gradients back to zero
                self.gen_op.zero_grad()
                self.class_op.zero_grad()
                self.adver_op.zero_grad()

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
                    self.class_op.step()
                    self.gen_op.step()
                else:
                    train_loss = -self.model.get_loss(
                        rec_loss, class_loss, adv_loss, label_y)
                    # backpropagate the gradient encoder-decoder-classifier with fixed adversary
                    train_loss.backward()
                    self.adver_op.step()

                U = not U

                # train the adversary
                # for t in range(2):
                #adv_loss = self.train_adversary_on_batch( train_data, sensitive_a, label_y)

                loss_log += train_loss.item()
                clas_loss_log += class_loss.item()
                rec_loss_log += rec_loss.item()
                adv_loss_log += adv_loss.item()
                # print(train_loss)
                # if(n_batch) % 100 == 0:
                #print("epoch : {}/{}, batch = {}, loss = {:.6f}".format(epoch + 1, num_epochs, n_batch, loss_log))
            # epoch loss
            loss_log = loss_log / len(self.data)
            rec_loss_log = rec_loss_log / len(self.data)
            adv_loss_log = adv_loss_log / len(self.data)
            clas_loss_log = clas_loss_log / len(self.data)
            self.logger.log(rec_loss_log, clas_loss_log,
                            adv_loss_log, epoch, num_epochs, len(self.data))
            # display the epoch training loss
            print("epoch : {}/{}, loss = {:.6f}, adv_loss:{:.6f}, class_loss:{:.6f}, rec_loss:{:.6f}".format(
                epoch + 1, num_epochs, loss_log, adv_loss_log, clas_loss_log, rec_loss_log))
        #self.logger.save_model(self.model.autoencoder, self.name)
        self.logger.close()

    def train(self, num_epochs=1000):
        """Train with fixed adversary or classifier-encoder-decoder across epoch

        Args:
            num_epochs (int, optional): [description]. Defaults to 1000.
        """
        for epoch in range(num_epochs):  # loop over dataset
            adv_loss_log = 0
            loss_log = 0
            clas_loss_log = 0
            rec_loss_log = 0
            for n_batch, (train_x, label_y, sensitive_a) in enumerate(self.data):
                train_data = Variable(train_x)

                train_data = train_data.to(device)
                label_y = label_y.to(device)
                sensitive_a = sensitive_a.to(device)

                # reset the gradients back to zero
                self.gen_op.zero_grad()
                self.class_op.zero_grad()

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
                with torch.no_grad():
                    adv_loss = self.model.get_adv_loss(pred_a, sensitive_a)

                # compute the classification loss
                class_loss = self.model.get_class_loss(pred_y, label_y)

                # compute the recontruction loss
                rec_loss = self.model.get_recon_loss(
                    reconstructed, train_data)
                train_loss = self.model.get_loss(
                    rec_loss, class_loss, adv_loss, label_y)

                # backpropagate the gradient encoder-decoder-classifier with fixed adversary
                train_loss.backward()

                # update parameter of the classifier and the autoencoder
                self.class_op.step()
                self.gen_op.step()

                # train the adversary
                for t in range(10):
                    #print("update adversary iter=", t)
                    adv_loss = self.train_adversary_on_batch(
                        train_data, sensitive_a, label_y, t)

                loss_log += train_loss.item()
                clas_loss_log += class_loss.item()
                rec_loss_log += rec_loss.item()
                adv_loss_log += adv_loss.item()
                # print(train_loss)
                # if(n_batch) % 100 == 0:
                #print("epoch : {}/{}, batch = {}, loss = {:.6f}".format(epoch + 1, num_epochs, n_batch, loss_log))

            else:
                # test error
                #print(sensitive_a, pred_a)
                #self.test(test_data_loader, num_epochs, epoch, n_batch)
                pass
            # epoch loss
            loss_log = loss_log / len(self.data)
            rec_loss_log = rec_loss_log / len(self.data)
            adv_loss_log = adv_loss_log / len(self.data)
            clas_loss_log = clas_loss_log / len(self.data)
            self.logger.log(rec_loss_log, clas_loss_log,
                            adv_loss_log, epoch, num_epochs, len(self.data))
            # display the epoch training loss
            print("epoch : {}/{}, loss = {:.6f}, adv_loss:{:.6f}, class_loss:{:.6f}, rec_loss:{:.6f}".format(
                epoch + 1, num_epochs, loss_log, adv_loss_log, clas_loss_log, rec_loss_log))
        #self.logger.save_model(self.model.autoencoder, self.name)
        self.logger.close()

    def test(self, test_data_loader, num_epochs, epoch, n_batch):
        adv_test_loss = 0
        class_test_loss = 0
        rec_test_loss = 0
        test_loss = 0
        model = self.model
        with torch.no_grad():
            for n_batch, (train_x, label_y, sensitive_a) in enumerate(test_data_loader):
                train_data, label_y, sensitive_a = Variable(
                    train_x), Variable(label_y), Variable(sensitive_a)

                if torch.cuda.is_available():
                    train_data = train_data.to(device)
                    label_y = label_y.cuda()
                    sensitive_a = sensitive_a.cuda()

                # compute reconstruction and latent space
                reconstructed, z = model.autoencoder(train_data)

                # predict class label from Z
                pred_y = model.classifier(z)

                adv_input = z
                if isinstance(model, EqualOddModel):
                    adv_input = torch.cat(
                        (z, label_y.view(label_y.shape[0], 1)), 1)
                # predict sentive attribut from Z
                pred_a = model.adversary(adv_input)  # fixed adversary

                # compute the recontruction loss
                rec_test_loss += model.get_recon_loss(
                    reconstructed, train_data)

                # compute the classification loss
                class_test_loss += model.get_class_loss(
                    pred_y, label_y.unsqueeze(1))

                # compute the adversary loss
                adv_test_loss += model.get_adv_loss(pred_a,
                                                    sensitive_a.unsqueeze(1))

                #test_loss += self.model.get_loss(rec_test_loss, class_test_loss, adv_test_loss, label_y).item()
            # test_loss = test_loss / len(test_data_loader) #model.get_loss(rec_lthe Equal Opportunity fairness constraint (Hardt, Price, andSrebro 2016) combined with ERM will provably recover the Bayes Optimal Classifierunder a range of bias modelsoss, class_loss, adv_loss, label_y)
            adv_test_loss = adv_test_loss.item() / len(test_data_loader)
            class_test_loss = class_test_loss.item() / len(test_data_loader)
            rec_test_loss = rec_test_loss.item() / len(test_data_loader)
            self.logger.log(rec_test_loss, class_test_loss, adv_test_loss,
                            epoch, n_batch, len(test_data_loader), description='test')
            #print("test_epoch : {}/{}, loss:{:.6f} , adv_loss:{:.6f}, class_loss:{:.6f}, rec_loss:{:.6f}".format(epoch + 1, num_epochs, test_loss, rec_test_loss, class_test_loss, rec_test_loss))


def train_classifier(classifier, params, is_avd=False):
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
        logger.save_model(classifier, model_name)
