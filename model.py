import torch
from torch import nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from utils import Logger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fn_rec_criteria = nn.MSELoss()
fn_bce_criteria = nn.BCELoss()


class AbstractModel(ABC):
    def __init__(self, n_feature, latent_dim, y_dim=1, recon_weight=1, class_weight=1, adv_weight=1, hidden_layers={}):
        self.recon_weight = recon_weight
        self.class_weight = class_weight
        self.adv_weight = adv_weight
        self.n_feature = n_feature
        self.latent_dim = latent_dim
        self.y_dim = y_dim
        self.name = "base"
        self.hidden_layers = hidden_layers

        self.autoencoder = AutoEncoder(
            n_feature=n_feature, latent_dim=latent_dim, hidden_layer=[hidden_layers['ae']])

        self.adversary = MLP(input_dim=latent_dim,
                             out_dim=y_dim, hidden_layer=hidden_layers['avd'])
        self.classifier = MLP(
            input_dim=latent_dim, out_dim=y_dim, hidden_layer=hidden_layers['class'])

        self.autoencoder = self.autoencoder.double().to(device)
        self.adversary = self.adversary.double().to(device)
        self.classifier = self.classifier.double().to(device)

    def add_stack_layer(self, laten_dim=8, layer_name='stack1', freeze_previous_layers=False):

        if freeze_previous_layers:
            for param in self.autoencoder.parameters():
                param.requires_grad = False
            # for param in self.adversary.parameters():
            #    param.requires_grad = False

        self.autoencoder.add_stack_layer(
            laten_dim, layer_name)

        # define new adversary and classifiers
        self.adversary = MLP(input_dim=laten_dim,
                             out_dim=self.y_dim, hidden_layer=self.hidden_layers['avd'])
        self.adversary = self.adversary.double().to(device)
        self.classifier = MLP(
            input_dim=laten_dim, out_dim=self.y_dim, hidden_layer=self.hidden_layers['class'])
        self.classifier = self.classifier.double().to(device)

        self.latent_dim = laten_dim

    def get_mapping_function(self):
        classifier = self.autoencoder.encoder
        for param in classifier.parameters():
            param.requires_grad = False
        # add the classifier layer
        classifier_layer = nn.Sequential(
            nn.Linear(self.latent_dim, 1),
            nn.Sigmoid()
        )
        classifier.to(device)
        classifier.add_module("classifier", classifier_layer)
        return classifier.double()

    @abstractmethod
    def get_adv_loss(self, a_pred, a):
        pass

    @abstractmethod
    def get_recon_loss(self, x_prim, x):
        pass

    @abstractmethod
    def get_class_loss(self, y_pred, y):
        pass

    @abstractmethod
    def get_loss(self, recon_loss, class_loss, adv_loss, Y=None):
        pass


class DemParModel(AbstractModel):
    """
        Model that implement statistical parity
    """

    def __init__(self, n_feature, latent_dim, hidden_layers, recon_weight=1, class_weight=1, adv_weight=1):
        AbstractModel.__init__(self, n_feature=n_feature, latent_dim=latent_dim,
                               hidden_layers=hidden_layers, recon_weight=recon_weight, class_weight=class_weight, adv_weight=adv_weight)
        self.name = "Dem_Par"

    def get_adv_loss(self, a_pred, a):
        return fn_bce_criteria(a_pred, a)

    def get_recon_loss(self, x_prim, x):
        return fn_rec_criteria(x_prim, x)

    def get_class_loss(self, y_pred, y):
        return fn_bce_criteria(y_pred, y)

    def get_loss(self, recon_loss, class_loss, adv_loss, Y=None):
        loss = self.recon_weight*recon_loss + self.class_weight*class_loss + self.adv_weight*adv_loss
        return loss

    def transform(self, data):
        return self.autoencoder.encoder(data)


class EqualOddModel(DemParModel):
    """ For equalized odds, the label Y is passed to adversary to upper bound the equalized odds metric """

    def __init__(self, n_feature, latent_dim, hidden_layers, recon_weight=1, class_weight=1, adv_weight=1):
        DemParModel.__init__(self, n_feature=n_feature, latent_dim=latent_dim,
                             hidden_layers=hidden_layers, recon_weight=recon_weight, class_weight=class_weight, adv_weight=adv_weight)
        self.adversary = MLP(input_dim=self.latent_dim + self.y_dim, out_dim=self.y_dim,
                             hidden_layer=hidden_layers['avd'])  # for equalized odds and equal opportunity

        if torch.cuda.is_available():
            self.adversary = self.adversary.double().cuda()
        else:
            self.adversary = self.adversary.double()
        self.name = "Eq_Odds"

    def add_stack_layer(self, laten_dim=8, layer_name='stack1', freeze_previous_layers=False):
        if freeze_previous_layers:
            for param in self.autoencoder.parameters():
                param.requires_grad = False

        self.autoencoder.add_stack_layer(
            laten_dim, layer_name)

        # self.classifier.add_input_layer(laten_dim)

        # self.adversary.add_input_layer(laten_dim+1)
        self.adversary = MLP(input_dim=laten_dim + self.y_dim, out_dim=self.y_dim,
                             hidden_layer=self.hidden_layers['avd'])
        self.adversary = self.adversary.double()

        self.classifier = MLP(
            input_dim=laten_dim, out_dim=self.y_dim, hidden_layer=self.hidden_layers['class'])
        self.classifier = self.classifier.double().to(device)

        self.latent_dim = laten_dim


class EqualOppModel(DemParModel):
    def __init__(self, n_feature, latent_dim, hidden_layers, recon_weight=1, class_weight=1, adv_weight=1):
        DemParModel.__init__(self, n_feature=n_feature, latent_dim=latent_dim,
                             hidden_layers=hidden_layers, recon_weight=recon_weight, class_weight=class_weight, adv_weight=adv_weight)
        self.name = "Eq_Opp"

    def get_loss(self, recon_loss, class_loss, adv_loss, Y=None):
        """ Similare to DemParModel but with Y = 0, this will enfore P(Y^=1|S, Y=1)"""
        loss = self.recon_weight*recon_loss + self.class_weight *class_loss + self.adv_weight*adv_loss
        if Y != None:
            loss = torch.multiply(1-Y, loss)
        return loss


class AutoEncoder(nn.Module):
    def __init__(self, n_feature, latent_dim, hidden_layer=[8]):
        super(AutoEncoder, self).__init__()
        self.n_feature = n_feature
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.n_feature, hidden_layer[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layer[-1], self.latent_dim),
            # nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_layer[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layer[-1], self.n_feature),
            # nn.ReLU(),
        )

    def add_stack_layer(self, laten_dim=8, layer_name='stack1'):
        new_layer = nn.Sequential(
            nn.Linear(self.latent_dim, laten_dim),
            nn.ReLU()
        ).double()

        new_deco_layer = nn.Sequential(
            nn.Linear(laten_dim, self.latent_dim),
            nn.ReLU()
        ).double()

        new_layer = new_layer.to(device)
        new_deco_layer = new_deco_layer.to(device)

        self.encoder.add_module(layer_name, new_layer)
        new_deco_layer.add_module(layer_name, self.decoder)
        self.decoder = new_deco_layer
        self.latent_dim = laten_dim

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z


class MLP(nn.Module):
    """
        A multi-layer feed forward network     
    """

    def __init__(self, input_dim=8, out_dim=1, hidden_layer=20):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, self.out_dim)
        )
        #self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.sigmoid = nn.Sigmoid()

    def add_input_layer(self, input_dim, layer_name='stack1'):
        new_layer = nn.Sequential(
            nn.Linear(input_dim, self.input_dim),
            nn.ReLU()
        ).double()

        self.input_dim = input_dim

        new_layer.to(device)

        new_layer.add_module(layer_name, self.net)
        self.net = new_layer

    def predict(self, x):
        # This function takes an input and predicts the class label, (0 or 1)
        pred = self.forward(x)
        #ans = torch.round(torch.sigmoid(pred))
        ans = torch.round(pred)
        return ans  # torch.tensor(ans, dtype=torch.long)

    def forward(self, x):
        x = self.net(x)
        x = self.sigmoid(x)
        return x.squeeze(1)


def cross_entropy(y_pred, y):
    """
    Calculate the mean cross entropy.
        y: expected class labels.
        y]: predicted class scores. 
    Return: the cross entropy loss. 
    """
    return -torch.mean(torch.mul(y_pred, torch.log(y)) + torch.mul((1-y_pred), torch.log(1-y)))
