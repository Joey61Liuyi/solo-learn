import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision.models import resnet18, resnet50
import torch.nn as nn
from solo.methods.linear import LinearModel  # imports the linear eval class
from solo.data.classification_dataloader import prepare_data
from solo.utils.checkpointer import Checkpointer

kwargs = {
    "num_classes": 10,
    "cifar": True,
    "max_epochs": 100,
    "optimizer": "sgd",
    "precision": 16,
    "lars": False,
    "lr": 0.1,
    "exclude_bias_n_norm_lars": False,
    "gpus": "0",
    "weight_decay": 0,
    "extra_optimizer_args": {"momentum": 0.9},
    "scheduler": "step",
    "lr_decay_steps": [60, 80],
    "batch_size": 128,
    "num_workers": 4,
    "pretrained_feature_extractor": "cifar_10_trained_models_batch32_32_noise/simclr/w3nll2eu/simclr-cifar10-w3nll2eu-ep=999.ckpt"
}

method = 'simclr'
dataset = 'cifar10'

# if method == 'byol':
#     kwargs['pretrained_feature_extractor'] = "trained_models/byol/1e7w3aum/byol-cifar100-1e7w3aum-ep=999.ckpt"
# elif method == 'simclr':
#     kwargs['pretrained_feature_extractor'] = "trained_models/simclr/5y61l57w/simclr-cifar100-5y61l57w-ep=999.ckpt"

backbone = resnet18()
backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
backbone.maxpool = nn.Identity()
backbone.fc = nn.Identity()

# load pretrained feature extractor
state = torch.load(kwargs["pretrained_feature_extractor"])["state_dict"]
for k in list(state.keys()):
    if "backbone" in k:
        state[k.replace("backbone.", "")] = state[k]
    del state[k]
backbone.load_state_dict(state, strict=False)

train_loader, val_loader = prepare_data(
    dataset,
    batch_size=kwargs["batch_size"],
    num_workers= 1,
)

prefix = 'noise_only'

with torch.no_grad():

    feature_train = []
    label_train = []

    for image, label in train_loader:
        feature = backbone(image)
        feature_train.append(feature)
        label_train.append(label)

    feature_train = torch.cat(feature_train, dim=0)
    label_train = torch.cat(label_train, dim=0)
    torch.save(feature_train, '{}_{}_{}_train_feature.pt'.format(prefix, method, dataset))
    torch.save(label_train, '{}_{}_{}_train_label.pt'.format(prefix, method, dataset))

    feature_eval = []
    label_eval = []

    for image, label in val_loader:
        feature = backbone(image)
        feature_eval.append(feature)
        label_eval.append(label)

    feature_eval = torch.cat(feature_eval, dim=0)
    label_eval = torch.cat(label_eval, dim=0)
    torch.save(feature_eval, '{}_{}_{}_eval_feature.pt'.format(prefix, method, dataset))
    torch.save(label_eval, '{}_{}_{}_eval_label.pt'.format(prefix, method, dataset))

print('ok')


# model = LinearModel(backbone, kwargs)

#
#
# wandb_logger = WandbLogger(
#     name="linear-cifar100",  # name of the experiment
#     project="self-supervised",  # name of the wandb project
#     entity=None,
#     offline=False,
# )
#
# wandb_logger.watch(model, log="gradients", log_freq=100)
#
# callbacks = []
#
# # automatically log our learning rate
# lr_monitor = LearningRateMonitor(logging_interval="epoch")
# callbacks.append(lr_monitor)
#
# # checkpointer can automatically log your parameters,
# # but we need to wrap them in a Namespace object
# from argparse import Namespace
# args = Namespace(**kwargs)
# # saves the checkout after every epoch
# ckpt = Checkpointer(
#     args,
#     logdir="checkpoints/linear",
#     frequency=1,
# )
# callbacks.append(ckpt)
#
# trainer = Trainer.from_argparse_args(
#     args,
#     logger=wandb_logger if args.wandb else None,
#     callbacks=callbacks,
#     checkpoint_callback=False,
#     terminate_on_nan=True,
# )
#
# trainer.fit(model, train_loader, val_loader)