# 训练 RESNET18，得到被攻击模型

from torchvision.datasets import cifar
import torchvision
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import resnet
import time

testmode = 1

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=50, device="cpu"):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        if testmode == 0:
            for batch in train_loader:
                optimizer.zero_grad()
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                output = model(inputs)
                loss = loss_fn(output, targets)
                loss.backward()
                optimizer.step()
                training_loss += loss.data.item() * inputs.size(0)
            training_loss /= len(train_loader.dataset)

            model.eval()
            num_correct = 0
            num_examples = 0
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                output = model(inputs)
                targets = targets.to(device)
                loss = loss_fn(output, targets)
                valid_loss += loss.data.item() * inputs.size(0)
                correct = torch.eq(torch.max(output, dim=1)[1], targets)
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
            valid_loss /= len(val_loader.dataset)

            print(
                'Epoch: {}, Training Loss: {:.2f}, '
                'Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss, valid_loss, num_correct / num_examples))
        else:
            model.eval()
            num_correct = 0
            num_examples = 0
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                output = model(inputs)
                targets = targets.to(device)
                loss = loss_fn(output, targets)
                valid_loss += loss.data.item() * inputs.size(0)
                correct = torch.eq(torch.max(output, dim=1)[1], targets)
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
            valid_loss /= len(val_loader.dataset)

            print(
                'Epoch: {}, Training Loss: {:.2f}, '
                'Validation Loss: {:.2f}, accuracy = {:.3f}'.format(epoch, training_loss, valid_loss,
                                                                    num_correct / num_examples))
            # print("sleeping...")
            # time.sleep(90)
            # print("awake")


def arangeDataset(dataset, arrageRatio=(0.6,0.1,0.6), batchsize=32):
    train_size = int(arrageRatio[0] * len(dataset))
    val_size = int(arrageRatio[1] * len(dataset))
    test_size = int(arrageRatio[2] * len(dataset))

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batchsize)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=batchsize)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=batchsize)
    return (train_dataloader, val_dataloader, test_dataloader)



picTrans = torchvision.transforms.Compose([
        # torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
        #                     std=(0.2023, 0.1994, 0.2010))
    ])

batchsize = 128
TrainSet = cifar.CIFAR10('./', train=True, download=False, transform=picTrans)    # LOAD 60000 SAMPLES

device = torch.device('cuda')
# model = torchvision.models.resnet18(pretrained=False).to(device)
model = torch.load('./model/RESNET18.pth')
# model = resnet.ResNet18().to(device)

criteon = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4)
train_dataloader, val_dataloader,_ = arangeDataset(TrainSet, arrageRatio=(0.1, 0.8, 0.1), batchsize=128)
train(model, optimizer, criteon, train_dataloader, val_dataloader, epochs=10, device="cuda")

# torch.save(model, './model/RESNET18.pth')



