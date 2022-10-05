import torch
from torch import nn
from abc import ABC, abstractmethod
import use_cuda


use_cuda = use_cuda.use_cuda
device_name = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
device = torch.device(device_name)
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

        self.autoencoder = self.autoencoder.float().to(device)
        self.adversary = self.adversary.float().to(device)
        self.classifier = self.classifier.float().to(device)

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
                               hidden_layers=hidden_layers, recon_weight=recon_weight, class_weight=class_weight,
                               adv_weight=adv_weight)
        self.name = "Dem_Par"

    def get_adv_loss(self, a_pred, a):
        return fn_bce_criteria(a_pred, a)

    def get_recon_loss(self, x_prim, x):
        return fn_rec_criteria(x_prim, x)

    def get_class_loss(self, y_pred, y):
        return fn_bce_criteria(y_pred, y)

    def get_loss(self, recon_loss, class_loss, adv_loss, Y=None):
        loss = self.recon_weight * recon_loss + self.class_weight * class_loss + self.adv_weight * adv_loss
        return loss

    def transform(self, data):
        return self.autoencoder.encoder(data)


class EqualOddModel(DemParModel):
    """ For equalized odds, the label Y is passed to adversary to upper bound the equalized odds metric """

    def __init__(self, n_feature, latent_dim, hidden_layers, recon_weight=1, class_weight=1, adv_weight=1):
        DemParModel.__init__(self, n_feature=n_feature, latent_dim=latent_dim,
                             hidden_layers=hidden_layers, recon_weight=recon_weight, class_weight=class_weight,
                             adv_weight=adv_weight)
        self.adversary = MLP(input_dim=self.latent_dim + self.y_dim, out_dim=self.y_dim,
                             hidden_layer=hidden_layers['avd'])  # for equalized odds and equal opportunity

        '''if torch.cuda.is_available() and use_cuda:
            self.adversary = self.adversary.float().cuda()
        else:'''
        self.adversary = self.adversary.float().to(device)
        self.name = "Eq_Odds"


class EqualOppModel(DemParModel):
    def __init__(self, n_feature, latent_dim, hidden_layers, recon_weight=1, class_weight=1, adv_weight=1):
        DemParModel.__init__(self, n_feature=n_feature, latent_dim=latent_dim,
                             hidden_layers=hidden_layers, recon_weight=recon_weight, class_weight=class_weight,
                             adv_weight=adv_weight)
        self.name = "Eq_Opp"

    def get_loss(self, recon_loss, class_loss, adv_loss, Y=None):
        """ Similar to DemParModel but with Y = 0, this will enfore P(Y^=1|S, Y=1)"""
        loss = self.recon_weight * recon_loss + self.class_weight * class_loss + self.adv_weight * adv_loss
        if Y != None:
            loss = torch.multiply(1 - Y, loss)
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
        ).float()

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_layer[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layer[-1], self.n_feature),
            # nn.ReLU(),
        ).float()

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
        ).float()
        # self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.sigmoid = nn.Sigmoid()

    def predict(self, x):
        # This function takes an input and predicts the class label, (0 or 1)
        pred = self.forward(x)
        # ans = torch.round(torch.sigmoid(pred))
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
    return -torch.mean(torch.mul(y_pred, torch.log(y)) + torch.mul((1 - y_pred), torch.log(1 - y)))
