import torch.utils.data as Data
import torch

class MyData(Data.Dataset):
    # train_file = 'training.pt'
    # test_file = 'test.pt'

    def __init__(self, dataset, transform=None):
        self.transform = transform

        # if not self._check_exists():  # 检查文件在不在
        #     raise RuntimeError('Dataset not found.')
        # if train:
        #     data_file = self.train_file
        # else:
        #     data_file = self.test_file
        self.data ,self.targets= dataset


    def __getitem__(self, index):  # 给下标，然后返回该下标对应的item
        img, target = self.data[index], self.targets[index] #img 是tensor类型,shape [28,28]
        # img = Image.fromarray(img.numpy())  # 从一个numpy对象转换成一个PIL image 对象
        # if self.transform is not None:
        #     img = self.transform(img)#img 的size为[1,28,28]
        return img,target

    def __len__(self):  # 返回数据的长度
        return len(self.data)

if __name__ == "__main__":
    cln_dataset = torch.load("./cln_example.pt")
    adv_dataset = torch.load("./adv_mixed_example.pt")

    data = torch.cat((cln_dataset, adv_dataset),0)

    cln_label = torch.zeros(cln_dataset.shape[0], dtype=int)
    adv_label = torch.zeros(adv_dataset.shape[0], dtype=int) + 1

    label = torch.cat( (cln_label, adv_label), 0)
    dataset = [data, label]

    dataset = MyData(dataset)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=5)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=5)
    for batchdata in train_dataloader:
        inputs, targets = batchdata