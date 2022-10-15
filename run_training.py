import argparse
from dataset import Dataset
from model import DemParModel
from trainer import Trainer
from utils import gen_dataclasses


def get_parser():
    parser = argparse.ArgumentParser(description="Benchmarking Differentially Private Fair Deep Learning")
    parser.add_argument("--dataset", default="Adult",
                        help="Dataset for training (default: Adult)")
    parser.add_argument("--data_dir", default="dataset",
                        help="Directory for dataset (default: dataset)")
    parser.add_argument("--sensattr", type=str, default="sex",
                        help="Sensitive attribute (default: sex)")
    parser.add_argument("--age", type=int, default=65,
                        help="Age value if sensattr is age (default: 65)")
    parser.add_argument("--batch", type=int, default=1024,
                        help="Batch size (default: 1024)")
    parser.add_argument("--only_download_data", default=False, action="store_true",
                        help="Use just for download dataset (default: False)")

    parser.add_argument("--n_features", type=int,
                        help="Number of features in input")
    parser.add_argument("--n_classes", type=int,
                        help="Number of classes")
    parser.add_argument("--edepth", type=int, default=2,
                        help="Encoder MLP depth as in depth*[width] (default: 2)")
    parser.add_argument("--ewidths", type=int, default=32,
                        help="Encoder MLP width (default: 32)")
    parser.add_argument("--cdepth", type=int, default=2,
                        help="Classifier MLP depth as in depth*[width] (default: 2)")
    parser.add_argument("--cwidths", type=int, default=32,
                        help="Classifier MLP width (default: 32)")
    parser.add_argument("--adepth", type=int, default=2,
                        help="Adversary MLP depth as in depth*[width] (default: 2)")
    parser.add_argument("--awidths", type=int, default=32,
                        help="Adversary MLP width (default: 32)")
    parser.add_argument("--zdim", type=int, default=16,
                        help="All MLPs has this as input or output (default: 16)")
    parser.add_argument("--classweight", type=float, default=1.,
                        help="Weight of classification in total loss (default: 1)")
    parser.add_argument("--advweight", type=float, default=1.,
                        help="Weight of adversary in total loss (default: 1)")
    parser.add_argument("--aeweight", type=float, default=0.,
                        help="Weight of autoencoder in total loss (default: 0)")
    parser.add_argument("--activ_ae", type=str, default="leakyrelu",
                        help="Activation function in hiddens in autoencoder (default: leakyrelu)")
    parser.add_argument("--activ_adv", type=str, default="leakyrelu",
                        help="Activation function in hiddens in adversary (default: leakyrelu)")
    parser.add_argument("--activ_class", type=str, default="leakyrelu",
                        help="Activation function in hiddens in classifier (default: leakyrelu)")
    parser.add_argument("--e_activ_ae", type=str, default="sigmoid",
                        help="Activation function in the end of autoencoder (default: sigmoid)")
    parser.add_argument("--e_activ_adv", type=str, default="sigmoid",
                        help="Activation function in the end of adversary (default: sigmoid)")
    parser.add_argument("--e_activ_class", type=str, default="sigmoid",
                        help="Activation function in the end of classifier (default: sigmoid)")
    parser.add_argument("--no_cuda", default=False, action="store_true",
                        help="Don't use cuda (default: False)")
    parser.add_argument("--xavier", default=False, action="store_true",
                        help="Use Xavier initialisation (default: False)")
    parser.add_argument("--grad_clip_ae", type=float, default=1.,
                        help="Gradient norm clipping without privacy in autoencoder, use 0 for disabling (default: 1)")
    parser.add_argument("--grad_clip_adv", type=float, default=1.,
                        help="Gradient norm clipping without privacy in adversary, use 0 for disabling (default: 1)")
    parser.add_argument("--grad_clip_class", type=float, default=1.,
                        help="Gradient norm clipping without privacy in classifier, use 0 for disabling (default: 1)")

    parser.add_argument("--delta", type=float,
                        help="The target δ of the (ϵ,δ)-differential privacy guarantee")
    parser.add_argument("--eps", type=float, default=10.,
                        help="The target ϵ of the (ϵ,δ)-differential privacy guarantee (default: 10)")
    parser.add_argument("--max_grad_norm", type=float, default=1.,
                        help="The maximum L2 norm of per-sample gradients (default: 1)")
    parser.add_argument("--privacy_in", action="append",
                        help="Inject privacy in [name]")

    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--adv_on_batch", type=int, default=1,
                        help="Iterations of training adversary (default: 1)")
    parser.add_argument("--eval_step_fair", type=int, default=3,
                        help="Evaluate fairness each [eval_step_fair] epoch (default: 3)")
    parser.add_argument("--epoch", type=int, default=80,
                        help="Number of epochs (default: 80)")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    name_and_args_dict = {"laftr_model_args":
                              ['edepth', 'ewidths', 'adepth', 'awidths', 'cdepth',
                               'cwidths', 'zdim', 'activ_ae', 'activ_adv', 'activ_class',
                               'e_activ_ae', 'e_activ_adv', 'e_activ_class',
                               'classweight', 'xavier', 'aeweight', 'advweight',
                               'n_features', 'n_classes', 'no_cuda'],
                          "dataset_args":
                              ['dataset', 'data_dir', 'batch', 'age', 'sensattr', 'only_download_data'],
                          "privacy_args":
                              ['privacy_in', 'delta', 'eps', 'max_grad_norm'],
                          "trainer_args":
                              ['epoch', 'seed', 'dataset', 'adv_on_batch', 'eval_step_fair',
                               'grad_clip_ae', 'grad_clip_adv', 'grad_clip_class']
                          }

    laftr_model_args, dataset_args, privacy_args, trainer_args = \
        gen_dataclasses(args.__dict__, name_and_args_dict)
    d = Dataset(dataset_args)
    if not dataset_args.only_download_data:
        train_dataloader, test_dataloader = d.get_dataloader()
        laftr_model_args.n_features = d.n_features()
        laftr_model_args.n_classes = d.n_classes()
        privacy_args.delta = 1 / d.dataset_size()

        laftr_model = DemParModel(laftr_model_args)

        trainer = Trainer(laftr_model, (train_dataloader, test_dataloader), trainer_args, privacy_args)
        trainer.train_process()
    else:
        d.download_data()


if __name__ == "__main__":
    main()
