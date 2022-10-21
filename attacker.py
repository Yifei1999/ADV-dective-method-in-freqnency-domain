import os
import torch
import advertorch.attacks as atk
import torch.nn as nn


# Construct a LinfPGDAttack adversary instance
torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
filename = "./model/RESNET18.pth"
model = torch.load(filename)
model.to(device)
model.eval()

# define the attacker algorithm
# reference:
#     :param predict: forward pass function.
#     :param loss_fn: loss function.
#     :param eps: maximum distortion.
#     :param eps_iter: attack step size.
#     :param nb_iter: number of iterations
#     :param clip_min: mininum value per input dimension.
#     :param clip_max: maximum value per input dimension.
attacker = {
    "GradientSign": atk.GradientSignAttack( #
        model, nn.CrossEntropyLoss(reduction="sum"),
        eps=0.015
    ),
    "L2BasicIter": atk.L2BasicIterativeAttack(
        model, nn.CrossEntropyLoss(reduction="sum"),
        eps=0.75, nb_iter=20, eps_iter=0.1
    ),
    "LinfBasicIter": atk.LinfBasicIterativeAttack(
        model, nn.CrossEntropyLoss(reduction="sum"),
        eps=0.015, nb_iter=5, eps_iter=0.01
    ),
    "PGD": atk.PGDAttack(
        model, nn.CrossEntropyLoss(reduction="sum"),
        eps=0.015, nb_iter=5, eps_iter=0.01
    ), # PGD
    "LinfPGD": atk.LinfPGDAttack( #
        model, nn.CrossEntropyLoss(reduction="sum"),
        eps=0.015, nb_iter=10, eps_iter=0.01
    ), # LinfPGD
    "L2PGD": atk.L2PGDAttack(
        model,nn.CrossEntropyLoss(reduction="sum"),
        eps=0.75, nb_iter=20, eps_iter=0.1,
    ), # L2PGD
    "L1PGD": atk.L1PGDAttack(
        model,nn.CrossEntropyLoss(reduction="sum"),
        eps=1.25, nb_iter=20, eps_iter=0.2,
        ),
    "SparseL1Descent": atk.SparseL1DescentAttack(
        model,nn.CrossEntropyLoss(reduction="sum"),
        eps=2.0, nb_iter=20, eps_iter=0.3
    ),
    "L2MomentumIterativeAttack":atk.L2MomentumIterativeAttack(
        model, nn.CrossEntropyLoss(reduction="sum"),
        eps=0.75, nb_iter=20, decay_factor=1., eps_iter=0.1
    ),
    "LinfMomentumIterativeAttack":atk.LinfMomentumIterativeAttack(
        model, nn.CrossEntropyLoss(reduction="sum"),
        eps=0.015, nb_iter=5, decay_factor=1., eps_iter=0.01
    )
}


if __name__=="__main__":
    pass
