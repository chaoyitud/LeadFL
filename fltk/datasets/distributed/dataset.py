from abc import abstractmethod
from typing import Any

# from fltk.util.arguments import Arguments
from fltk.util.config import Config
from fltk.util.log import getLogger


class DistDataset:
    train_sampler = None
    test_sampler = None
    train_dataset = None
    test_dataset = None
    train_loader = None
    test_loader = None
    mal_loader = None
    mal_sampler = None
    mal_dataset = None
    logger = getLogger(__name__)

    def __init__(self, args: Config):
        self.args = args

    def get_args(self):
        """
        Returns the arguments.

        :return: Arguments
        """
        return self.args

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    def get_train_sampler(self):
        return self.train_sampler

    def get_test_sampler(self):
        return self.test_sampler

    @abstractmethod
    def init_train_dataset(self):
        pass

    @abstractmethod
    def init_test_dataset(self):
        pass
