import torch

class MyIterator():
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.lim = dataset.shape[0]
        self.cur = 0

    def __iter__(self):
        return self

    def __next__(self):
        if (self.cur + self.batch_size) <= self.lim:
            self.cur += self.batch_size
            return self.dataset[(self.cur-self.batch_size):(self.cur) ]

        else:
            raise StopIteration


if __name__ ==  "__main__":
    dataset = torch.arange(1,10,1)
    print(dataset)
    dataloader = MyIterator(dataset ,2)
    for i_batch, batch_data in enumerate(dataloader):
        print(i_batch)
        print(batch_data)