import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import  mydataloader
import os
import resnet

import numpy as np
import os
import random
# seed=2018
# random.seed(seed)
# os.environ["PYTHONHASHSEED"] = str(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True

learning_epoch = 150
learning_rate = 0.0005
batch_size = 128
arch_name = "lumchannelallin_32"
print("epoch:{}, lr:{}".format(learning_epoch,learning_rate))

class SimpleNet(nn.Module):
    # 48BLOCK 双卷积层 pass  # lr=0.0001  rate:94%
    # def __init__(self):
    #     self.modelname = "Block arch1"
    #     super(SimpleNet, self).__init__()  # 8*8*9
    #     self.conv1 = torch.nn.Conv2d(48, 96, 3, padding=1)  # 8*8*32
    #     self.dp1 = nn.Dropout(p=0.1)
    #     self.conv2 = torch.nn.Conv2d(96, 192, 3, padding=1)  # 8*8*192
    #     self.pool1 = torch.nn.MaxPool2d(2, 2)  # 4*4*192
    #     self.conv3 = torch.nn.Conv2d(192, 192, 3, padding=1)  # 4*4*192
    #     self.pool2 = torch.nn.MaxPool2d(2, 2)  # 2*2*192
    #     self.fc1 = nn.Linear(2*2*192, 256)
    #     self.fc2 = nn.Linear(256, 48)
    #     self.fc3 = nn.Linear(48, 10)
    #     self.fc4 = nn.Linear(10, 2)
    #
    # def forward(self, x):
    #     x = F.relu(self.conv1(x))
    #     x = self.dp1(x)
    #     x = F.relu(self.conv2(x))
    #     x = self.pool1(x)
    #     x = F.relu(self.conv3(x))
    #     x = self.pool2(x)
    #     x = x.view(-1, 2*2*192)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = F.relu(self.fc3(x))
    #     x = F.relu(self.fc4(x))
    #     return x

    # 全通道
    def __init__(self): # epoch：60 ACC：97
        super(SimpleNet, self).__init__()  # 3*32*32
        self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.dp1 = nn.Dropout(p=0.2)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, padding=1)  # 32*32*16
        self.pool1 = torch.nn.MaxPool2d(2, 2)  # 16*16*16
        self.conv3 = torch.nn.Conv2d(16, 32, 3, padding=1)  # 4*4*192
        self.pool2 = torch.nn.MaxPool2d(2, 2)  # 8*8*32
        self.conv4 = torch.nn.Conv2d(32, 32, 3, padding=1)  # 4*4*32
        self.pool3 = torch.nn.MaxPool2d(2, 2)  # 4*4*32
        self.fc1 = nn.Linear(4*4*32, 128)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dp1(x)
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = F.relu(self.conv4(x))
        x = self.pool3(x)
        x = x.view(-1, 4*4*32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x




def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=64, device="cpu"):
    timer = 0
    for epoch in range(epochs):
        # timer += 1
        # if timer == 11:
        #     a = input("waiting...")
        #     if a == "p": break
        #     else: timer = 1

        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()    # take a step forward
            optimizer.step()    # take a step forward
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

def testbench(model, testloader):
    num_correct = 0
    num_examples = 0
    for batch in testloader:
        inputs, targets = batch
        inputs = inputs.to("cuda")
        output = model(inputs)
        targets = targets.to("cuda")
        correct = torch.eq(torch.max(output, dim=1)[1], targets)
        num_correct += torch.sum(correct).item()
        num_examples += correct.shape[0]

    print(" - correct: {}, total: {}, accuracy = {:.3f}".format(num_correct, num_examples,num_correct / num_examples))


def arangeDataset(dataset, arrageRatio=(0.9,0.09,0.01), batchsize=32):
    train_size = int(arrageRatio[0] * len(dataset))
    val_size = int(arrageRatio[1] * len(dataset))
    test_size = len(dataset)- val_size - train_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batchsize)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=batchsize)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=batchsize)
    return (train_dataloader, val_dataloader,test_dataloader)


if __name__ == "__main__":
    name_append = ""    # 频率域

    data_tensor = torch.load("./adv_sample/cln_remained_example" + name_append + ".pt")
    adv_tensor = torch.load("./adv_sample/adv_mixed_example" + name_append + ".pt")
    print("load {} clean samples".format(data_tensor.shape[0]))
    print("load {} adver samples".format(adv_tensor.shape[0]))
    data_tensor = data_tensor[0:25000]
    adv_tensor = adv_tensor[0:25000]

    # 数据处理： 亮度
    # data_tensor = (1.0*data_tensor[:,0,:,:] ).unsqueeze(1)
    # adv_tensor = (1.0 * adv_tensor[:,0,:, :] ).unsqueeze(1)


    data = torch.cat((data_tensor, adv_tensor), 0)

    # 0 for clean, 1 for adv
    label_tensor = torch.zeros(data_tensor.shape[0], dtype=torch.int64) + 0
    adv_label = torch.zeros(adv_tensor.shape[0], dtype=torch.int64) + 1
    label = torch.cat((label_tensor, adv_label), 0)
    dataset = [data, label]

    # generate dataloader
    dataset = mydataloader.MyData(dataset)
    train_dataloader, val_dataloader, test_dataloader = arangeDataset(dataset, arrageRatio=(0.7, 0.2, 0.1), batchsize=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    simplenet = SimpleNet()
    optimizer = optim.Adam(simplenet.parameters(), lr=learning_rate)
    simplenet.to(device)

    # using lossfunction: cross-entropy with softmax
    train(simplenet, optimizer, torch.nn.CrossEntropyLoss(), train_dataloader, val_dataloader, epochs=learning_epoch, device=device)
    torch.save(simplenet,"./model/"+arch_name+".pt")


    # test stage
    adv_testdata =  torch.load("./testdata/adv_mixed_example" + name_append + ".pt")
    cln_testdata = torch.load("./testdata/cln_remained_example" + name_append + ".pt")

    # adv_testdata = (1.0 * adv_testdata[:,0, :, :]).unsqueeze(1)
    # cln_testdata = (1.0 * cln_testdata[:, 0, :, :]).unsqueeze(1)

    adv_label = torch.zeros( adv_testdata.shape[0], dtype=torch.int64) + 1
    cln_label = torch.zeros(cln_testdata.shape[0], dtype=torch.int64) + 0

    advdataset = [adv_testdata, adv_label]
    clndataset = [cln_testdata, cln_label]

    adv_dataloader = mydataloader.MyData(advdataset)
    cln_dataloader = mydataloader.MyData(clndataset)

    adv_dataloader = torch.utils.data.DataLoader(adv_dataloader, shuffle=False, batch_size=32)
    cln_dataloader = torch.utils.data.DataLoader(cln_dataloader, shuffle=False, batch_size=32)

    print("on the AVDER testset:")
    testbench(simplenet, adv_dataloader)
    print("on the CLEAN testset:")
    testbench(simplenet, cln_dataloader)




# if __name__ == "__main__":  # 使用单一方法做测试
#     model = torch.load('./model/3channelallin_32.pt')
#     print(model)
#     path  = './singletest/small/freq_all/'
#     filename = os.listdir(path)
#
#     for name in filename:
#
#         print("read from " + name)
#         data_tensor = torch.load("./singletest/small/freq_all/"+name)
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#
#         label_tensor = torch.zeros(data_tensor.shape[0], dtype=torch.int64) + 1  # 对抗样本
#         dataset = [data_tensor, label_tensor]
#         dataset = mydataloader.MyData(dataset)
#         dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=32)
#         testbench(model, dataloader)
