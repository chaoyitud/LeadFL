# pylint: disable=missing-function-docstring,missing-class-docstring,invalid-name
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from fltk.datasets.distributed.dataset import DistDataset
from fltk.samplers import get_sampler
from fltk.datasets.distributed.get_mal_dataset import get_mal_dataset
from fltk.datasets.datasetsplit import DatasetSplit


class DistFashionMNISTDataset(DistDataset):

    def __init__(self, args):
        super(DistFashionMNISTDataset, self).__init__(args)
        self.init_train_dataset()
        self.init_test_dataset()

    def init_train_dataset(self):
        dist_loader_text = "distributed" if self.args.get_distributed() else ""
        self.logger.debug(f"Loading '{dist_loader_text}' Fashion MNIST train data")

        self.train_dataset = datasets.FashionMNIST(root=self.get_args().get_data_path(), train=True, download=True,
                                                   transform=transforms.Compose([transforms.ToTensor()]))
        self.train_sampler = get_sampler(self.train_dataset, self.args)
        self.train_loader = DataLoader(self.train_dataset, batch_size=16, sampler=self.train_sampler)

    def init_test_dataset(self):
        dist_loader_text = "distributed" if self.args.get_distributed() else ""
        self.logger.debug(f"Loading '{dist_loader_text}' Fashion MNIST test data")
        self.test_dataset = datasets.FashionMNIST(root=self.get_args().get_data_path(), train=False, download=True,
                                                  transform=transforms.Compose([transforms.ToTensor()]))
        #self.test_sampler = get_sampler(self.test_dataset, self.args)
        self.test_sampler = get_sampler(self.test_dataset, self.args, test=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.args.test_batch_size,
                                      sampler=self.test_sampler)

    def init_mal_dataset(self):
        dist_loader_text = "distributed" if self.args.get_distributed() else ""
        mal_X_list, mal_Y, Y_true = get_mal_dataset(self.test_dataset, self.args.mal_samples, 10)
        self.mal_loader = DataLoader(DatasetSplit(self.test_dataset, mal_X_list, mal_Y),
                                     batch_size=self.args.batch_size, shuffle=True)

    def get_mal_loaders(self):
        return self.mal_loader
