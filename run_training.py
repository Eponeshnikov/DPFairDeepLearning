import argparse
import torch
import numpy as np
from dataset import Dataset
from model import DemParModel, EqualOddModel
from trainer import Trainer
from utils import gen_dataclasses


def get_parser():
    parser = argparse.ArgumentParser(description="Benchmarking Differentially Private Fair Deep Learning")
    parser.add_argument("--dataset", type=str, default="Adult",
                        help="Dataset for training (default: Adult)")
    parser.add_argument("--data_dir", default="dataset",
                        help="Directory for dataset (default: dataset)")
    parser.add_argument("--sensattr", type=str, default="sex",
                        help="Sensitive attribute (default: sex)")
    parser.add_argument("--age_low", type=int, default=10,
                        help="Low bound of age value if sensattr is age (default: 10)")
    parser.add_argument("--age_high", type=int, default=65,
                        help="High bound of age value if sensattr is age (default: 65)")
    parser.add_argument("--batch", default=1024,
                        help="Batch size (default: 1024); set 'max' for using all data in one batch")
    parser.add_argument("--only_download_data", default=False, action="store_true",
                        help="Use just for download dataset (default: False)")

    parser.add_argument("--arch", type=str, default="DP",
                        help="Architecture of model")
    parser.add_argument("--n_features", type=int,
                        help="Number of features in input")
    parser.add_argument("--n_classes", type=int,
                        help="Number of classes")
    parser.add_argument("--n_groups", type=int,
                        help="Number of groups")
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
    parser.add_argument("--optimizer_enc_class", type=str, default='NAdam',
                        help="Encoder-Classifier optimizer (default: NAdam)")
    parser.add_argument("--optimizer_adv", type=str, default='NAdam',
                        help="Adversary optimizer (default: NAdam)")
    parser.add_argument("--enc_class_sch", type=str, default='PolynomialLR',
                        help="Encoder-Classifier scheduler (default: PolynomialLR)")
    parser.add_argument("--adv_sch", type=str, default='PolynomialLR',
                        help="Adversary scheduler (default: PolynomialLR)")
    parser.add_argument("--enc_class_sch_pow", type=float, default=1.5,
                        help="Power for PolynomialLR encoder-classifier scheduler (default: 1.5)")
    parser.add_argument("--adv_sch_pow", type=float, default=1.5,
                        help="Power for PolynomialLR adversary scheduler (default: 1.5)")
    parser.add_argument("--lr_enc_class", type=float, default=0.11,
                        help="Learning rate for encoder-classifier optimizer (default: 0.11)")
    parser.add_argument("--lr_adv", type=float, default=0.11,
                        help="Learning rate for adversary optimizer (default: 0.11)")

    parser.add_argument("--check_acc_fair", default=True, action="store_true",
                        help="Rerun experiment if last value of test accuracy < 0.5"
                             " and fair metrics > 0.01 (default: True)")
    parser.add_argument("--check_acc_fair_attempts", type=int, default=5,
                        help="Attempts for train model with --check_acc_fair (default: 5)")
    parser.add_argument("--acc_tresh", type=float, default=0.5,
                        help="Accuracy threshold for check_acc_fair (default: 0.5)")
    parser.add_argument("--dp_atol", type=float, default=0.02,
                        help="DP tolerance for check_acc_fair (default: 0.02)")
    parser.add_argument("--eod_atol", type=float, default=0.02,
                        help="EOD tolerance for check_acc_fair (default: 0.02)")
    parser.add_argument("--offline_mode", default=False, action="store_true",
                        help="Offline mode for ClearML (default: False)")
    parser.add_argument("--eval_model", type=str, default='LR',
                        help="Model for evaluation metrics (default: LR - LogisticRegression)")
    parser.add_argument("--config_dir", default="configs",
                        help="Directory for configs (default: configs)")
    parser.add_argument("--server", default="None",
                        help="Name of .conf file for ClearML server, use None for default (default: None)")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    name_and_args_dict = {"laftr_model_args":
                              ['arch', 'edepth', 'ewidths', 'adepth', 'awidths', 'cdepth', 'cwidths', 'zdim',
                               'activ_ae', 'activ_adv', 'activ_class', 'e_activ_ae', 'e_activ_adv', 'e_activ_class',
                               'classweight', 'aeweight', 'advweight', 'n_features', 'n_classes', 'n_groups', 'xavier',
                               'no_cuda', 'seed'],
                          "dataset_args":
                              ['dataset', 'data_dir', 'batch', 'age_low', 'age_high', 'sensattr', 'only_download_data',
                               'seed'],
                          "privacy_args":
                              ['privacy_in', 'delta', 'eps', 'max_grad_norm'],
                          "trainer_args":
                              ['epoch', 'seed', 'dataset', 'adv_on_batch', 'eval_step_fair', 'grad_clip_ae',
                               'grad_clip_adv', 'grad_clip_class', 'sensattr', 'optimizer_enc_class', 'optimizer_adv',
                               'lr_enc_class', 'lr_adv', 'check_acc_fair', 'enc_class_sch', 'adv_sch',
                               'enc_class_sch_pow', 'adv_sch_pow', 'eval_model', 'offline_mode',
                               'check_acc_fair_attempts', 'config_dir', 'server', 'acc_tresh', 'dp_atol', 'eod_atol']
                          }

    laftr_model_args, dataset_args, privacy_args, trainer_args = \
        gen_dataclasses(args.__dict__, name_and_args_dict)
    d = Dataset(dataset_args)
    if not dataset_args.only_download_data:
        device_name = "cuda" if torch.cuda.is_available() and not laftr_model_args.no_cuda else "cpu"
        train_dataloader, test_dataloader = d.get_dataloader(device_name)
        laftr_model_args.n_features = d.n_features()
        laftr_model_args.n_classes = d.n_classes()
        laftr_model_args.n_groups = d.n_groups()
        privacy_args.delta = 1 / d.dataset_size()
        if laftr_model_args.arch == 'DP':
            model_arch = DemParModel
        elif laftr_model_args.arch == 'EOD':
            model_arch = EqualOddModel
        else:
            raise Exception('Only DP and EOD available')

        acc = 0
        dp = 1
        eod = 1
        attempt = 0
        while any([acc <= trainer_args.acc_tresh,
                   not np.isclose(dp, 0, atol=trainer_args.dp_atol),
                   not np.isclose(eod, 0, atol=trainer_args.eod_atol)]):
            laftr_model = model_arch(laftr_model_args)
            trainer = Trainer(laftr_model, (train_dataloader, test_dataloader), trainer_args, privacy_args)
            acc, dp, eod = trainer.train_process()
            attempt += 1
            if attempt == trainer_args.check_acc_fair_attempts:
                trainer.logger.task.mark_failed()
                trainer.logger.task.close()
                print(f'Attempts ended. Accuracy: {round(acc, 2)}, DP: {round(dp, 3)}, EOD: {round(eod, 3)}')
                break
            if not trainer_args.check_acc_fair:
                print(f'Accuracy: {round(acc, 2)}, DP: {round(dp, 3)}, EOD: {round(eod, 3)} no checking')
                break
            trainer_args.seed += 1
            laftr_model_args.seed += 1
            if any([acc <= trainer_args.acc_tresh,
                   not np.isclose(dp, 0, atol=trainer_args.dp_atol),
                   not np.isclose(eod, 0, atol=trainer_args.eod_atol)]):
                print(f'Wrongly trained, retry. Accuracy: {round(acc, 2)}, DP: {round(dp, 3)}, EOD: {round(eod, 3)}')
                try:
                    if not trainer_args.offline_mode:
                        trainer.logger.task.reset(set_started_on_success=False, force=True)
                        trainer.logger.task.close()
                        trainer.logger.task.set_archived(archive=True)
                    else:
                        pass
                except Exception as e:
                    print(e)
    else:
        d.download_data()


if __name__ == "__main__":
    main()
