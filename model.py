import torch
from torch import nn
from itertools import chain
from abc import ABC, abstractmethod

fn_rec_criteria = nn.MSELoss()
fn_bce_criteria = nn.BCELoss()


class AbstractModel(ABC):
    def __init__(self, args):
        self.device_name = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.device = torch.device(self.device_name)
        self.name = 'base'

        self.classweight = args.classweight
        self.aeweight = args.aeweight
        self.advweight = args.advweight

        self.zdim = args.zdim
        self.xavier = args.xavier

        self.autoencoder = AutoEncoder(args)
        self.class_neurons = [args.zdim] + args.cdepth * [args.cwidths] + [args.n_classes - 1]
        self.adv_neurons = [args.zdim] + args.adepth * [args.awidths] + [args.n_groups - 1]

        self.adversary = MLP(self.adv_neurons, activ=args.activ_adv,
                             end_activ=args.e_activ_adv, xavier=args.xavier)
        self.classifier = MLP(self.class_neurons, activ=args.activ_class,
                              end_activ=args.e_activ_class, xavier=args.xavier)

        self.autoencoder = self.autoencoder.to(self.device)
        self.adversary = self.adversary.to(self.device)
        self.classifier = self.classifier.to(self.device)

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

    def __init__(self, args):
        AbstractModel.__init__(self, args)
        self.name = "Dem_Par"

    def get_adv_loss(self, a_pred, a):
        return fn_bce_criteria(a_pred, a)

    def get_recon_loss(self, x_prim, x):
        return fn_rec_criteria(x_prim, x)

    def get_class_loss(self, y_pred, y):
        return fn_bce_criteria(y_pred, y)

    def get_loss(self, recon_loss, class_loss, adv_loss, Y=None):
        loss = self.aeweight * recon_loss + self.classweight * class_loss + self.advweight * adv_loss
        return loss

    def transform(self, data):
        return self.autoencoder.encoder(data)


class EqualOddModel(DemParModel):
    """ For equalized odds, the label Y is passed to adversary to upper bound the equalized odds metric """

    def __init__(self, args):
        DemParModel.__init__(self, args)
        self.adv_neurons = [args.zdim + args.n_classes - 1] + args.adepth * [args.awidths] + [args.n_groups - 1]

        self.adversary = MLP(self.adv_neurons)  # for equalized odds and equal opportunity

        self.adversary = self.adversary.to(self.device)
        self.name = "Eq_Odds"


class EqualOppModel(DemParModel):
    def __init__(self, args):
        DemParModel.__init__(self, args)
        self.name = "Eq_Opp"

    def get_loss(self, recon_loss, class_loss, adv_loss, Y=None):
        """ Similar to DemParModel but with Y = 0, this will enforce P(Y^=1|S, Y=1)"""
        loss = self.aeweight * recon_loss + self.classweight * class_loss + self.advweight * adv_loss
        if Y is not None:
            loss = torch.multiply(1 - Y, loss)
        return loss


class AutoEncoder(nn.Module):
    def __init__(self, args):
        super(AutoEncoder, self).__init__()
        self.enc_neurons = [args.n_features] + args.edepth * [args.ewidths] + [args.zdim]
        self.dec_neurons = [args.zdim] + args.edepth * [args.ewidths] + [args.n_features]

        self.encoder = MLP(self.enc_neurons, activ=args.activ_ae, end_activ=args.e_activ_ae, xavier=args.xavier)
        self.decoder = MLP(self.dec_neurons, activ=args.activ_ae, end_activ=args.e_activ_ae, xavier=args.xavier)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z


class MLP(nn.Module):
    def __init__(self, num_neurons, activ="leakyrelu", end_activ='sigmoid', xavier=False):
        """Initializes MLP unit"""
        super(MLP, self).__init__()
        self.num_neurons = num_neurons
        self.activ = activ
        self.end_activ = end_activ
        self.num_layers = len(self.num_neurons) - 1
        self.activ_func = self.get_activ_func(self.activ)
        self.end_activ_func = self.get_activ_func(self.end_activ)

        self.hiddens = nn.Sequential(
            *[
                i for i in list(chain.from_iterable(
                    [
                        [nn.Linear(self.num_neurons[i], self.num_neurons[i + 1]), self.activ_func]
                        for i in range(self.num_layers)
                    ]
                ))[:-1] + [self.end_activ_func]
                if i is not None]
        )
        if xavier:
            for hidden in self.hiddens:
                if isinstance(hidden, nn.Linear):
                    torch.nn.init.xavier_uniform_(hidden.weight)

    def get_activ_func(self, activ):
        if activ == "softplus":
            activ_func = nn.Softplus()
        elif activ == "sigmoid":
            activ_func = nn.Sigmoid()
        elif activ == "relu":
            activ_func = nn.ReLU()
        elif activ == "leakyrelu":
            activ_func = nn.LeakyReLU()
        elif activ == "None":
            activ_func = None
        else:
            raise Exception("bad activation function")
        return activ_func

    def forward(self, x):
        """Computes forward pass through the model"""
        x = self.hiddens(x)
        return x.squeeze(1)

    def freeze(self):
        """Stops gradient computation through MLP parameters"""
        for para in self.parameters():
            para.requires_grad = False

    def activate(self):
        """Activates gradient computation through MLP parameters"""
        for para in self.parameters():
            para.requires_grad = True
