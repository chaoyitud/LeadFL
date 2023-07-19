from torch.utils.data import Dataset
import torch


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs, Y=None):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.mal = False
        if Y is not None:
            self.mal = True
            self.mal_Y = Y

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.mal == True:
            label_mal = self.mal_Y[item]
            return torch.tensor(image), torch.tensor(label_mal), torch.tensor(label)
        return torch.tensor(image), torch.tensor(label)
