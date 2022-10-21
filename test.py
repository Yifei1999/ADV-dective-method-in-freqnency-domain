from torch.utils.data import DataLoader
from torchvision.datasets import cifar
import torchvision.transforms as torchtrans

import torch
from advertorch.utils import predict_from_logits

TrainSet = cifar.CIFAR10('./', train=True, download=False, transform=torchtrans.ToTensor())
loader = DataLoader(TrainSet, batch_size=128, shuffle=True)
i=0
clean_sample = torch.tensor([]).to("cuda")
for data in loader:
    i=i+1
    if i >= 200:
        (cln_data, true_label) = data
        cln_data = cln_data.to("cuda")
        clean_sample = torch.cat((clean_sample, cln_data), 0)
        if i == 270: break

torch.save(clean_sample, "./adv_sample/cln_remained_example.pt")
print("1")