import torch
from torchvision.datasets import cifar
import torchvision.transforms as torchtrans

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
filename = "./model/RESNET18.pth"
model = torch.load(filename)
model.to(device)
model.eval()

# load data: number of test sample
batch_size = 128
selected_batch = 2
TrainSet = cifar.CIFAR10('./', train=True, download=False, transform=torchtrans.ToTensor())    # LOAD 60000 SAMPLES
train_dataloader = torch.utils.data.DataLoader(TrainSet, shuffle=True, batch_size=batch_size)

for i, batch in enumerate(train_dataloader):
    if i == selected_batch:
        cln_data, true_label = batch
        break

cln_data, true_label = cln_data.to(device), true_label.to(device)

# Construct a LinfPGDAttack adversary instance
from attacker import attacker

attacker_name = "L2MomentumIterativeAttack"  # list(attacker.keys())[1]
print("using method: " + attacker_name)
adversary = attacker[attacker_name]

# Perform untargeted attack
adv_untargeted = adversary.perturb(cln_data, true_label)

# Visualization of attacks
output = model(cln_data)
pred_cln = torch.max(model(cln_data), dim=1)[1]
pred_untargeted_adv = torch.max(model(adv_untargeted), dim=1)[1]

# count the result
correct = torch.eq(pred_untargeted_adv, pred_cln)
num_correct = torch.sum(correct).item()
num_examples = correct.shape[0]
print("attack failed rate: {}/{} = {}".format(num_correct, num_examples, num_correct / num_examples))

import matplotlib.pyplot as plt
import displayer

displayer.displayer(cln_data, display_row=3, display_col=5, batch_serialnum=6)
displayer.displayer(adv_untargeted, display_row=3, display_col=5, batch_serialnum=6)
displayer.displayer(0.5+(adv_untargeted-cln_data)*1, display_row=3, display_col=5, batch_serialnum=6)

plt.show()