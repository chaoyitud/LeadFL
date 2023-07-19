import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List, Type, Dict, T, Union, Any

import torch.distributed as dist
import torch.nn
from dataclasses_json import dataclass_json

from fltk.util.config.definitions import Nets, Dataset

_available_loss = {
    "CROSSENTROPYLOSS": torch.nn.CrossEntropyLoss,
    "HUBERLOSS" : torch.nn.HuberLoss
}

_available_optimizer: Dict[str, Type[torch.optim.Optimizer]] = {
    "SGD": torch.optim.SGD,
    "ADAM": torch.optim.Adam,
    "ADAMW": torch.optim.AdamW
}


@dataclass_json
@dataclass(frozen=True)
class DistLearningConfig:  # pylint: disable=too-many-instance-attributes
    """
    Class encapsulating LearningParameters, for now used under DistributedLearning.
    """
    model: Nets
    dataset: Dataset
    batch_size: int
    test_batch_size: int
    max_epoch: int
    learning_rate: float
    learning_decay: float
    loss: str
    optimizer: str
    optimizer_args: Dict[str, Any]
    scheduler_step_size: int
    scheduler_gamma: float
    min_lr: float

    cuda: bool
    seed: int

    @staticmethod
    def __safe_get(lookup: Dict[str, T], keyword: str) -> T:
        """
        Static function to 'safe' get elements from a dictionary, to prevent issues with Capitalization in the code.
        @param lookup: Lookup dictionary to 'safe get' from.
        @type lookup: dict
        @param keyword: Keyword to 'get' from the Lookup dictionary.
        @type keyword: str
        @return: Lookup value from 'safe get' request.
        @rtype: T
        """
        safe_keyword = str.upper(keyword)
        if safe_keyword not in lookup:
            logging.fatal(f"Cannot find configuration parameter {keyword} in dictionary.")
        return lookup.get(safe_keyword)

    # def get_model_class(self) -> Type[torch.nn.Module]:
    #     """
    #     Function to obtain the model class that was given via commandline.
    #     @return: Type corresponding to the model that was passed as argument.
    #     @rtype: Type[torch.nn.Module]
    #     """
    #     return get_net(self.model)

    # def get_dataset_class(self) -> Type[Dataset]:
    #     """
    #     Function to obtain the dataset class that was given via commandline.
    #     @return: Type corresponding to the dataset that was passed as argument.
    #     @rtype: Type[Dataset]
    #     """
    #     return get_dataset(self.dataset)

    def get_loss(self) -> Type:
        """
        Function to obtain the loss function Type that was given via commandline to be used during the training
        execution.
        @return: Type corresponding to the loss function that was passed as argument.
        @rtype: Type
        """
        return self.__safe_get(_available_loss, self.loss)

    def get_optimizer(self) -> Type[torch.optim.Optimizer]:
        """
        Function to obtain the loss function Type that was given via commandline to be used during the training
        execution.
        @return: Type corresponding to the Optimizer to be used during training.
        @rtype: Type[torch.optim.Optimizer]
        """
        return self.__safe_get(_available_optimizer, self.optimizer)

def _create_extractor_parser(subparsers):
    """
    Helper function to add extractor arguments.
    @param subparsers: Subparser to add arguments to.
    @type subparsers: Any
    @return: None
    @rtype: None
    """
    extractor_parser = subparsers.add_parser('extractor')
    extractor_parser.add_argument('config', type=str)


def _create_client_parser(subparsers) -> None:
    """
    Helper function to add client arguments.
    @param subparsers: Subparser to add arguments to.
    @type subparsers: Any
    @return: None
    @rtype: None
    """
    client_parser = subparsers.add_parser('client')
    client_parser.add_argument('config', type=str)
    client_parser.add_argument('task_id', type=str)
    client_parser.add_argument('experiment_config', type=str)
    # Add parameter parser for backend
    client_parser.add_argument('--backend', type=str, help='Distributed backend',
                               choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
                               default=dist.Backend.GLOO)


def _create_cluster_parser(subparsers) -> None:
    """
    Helper function to add cluster execution arguments.
    @param subparsers: Subparser to add arguments to.
    @type subparsers: Any
    @return: None
    @rtype: None
    """
    cluster_parser = subparsers.add_parser('cluster')
    cluster_parser.add_argument('config', type=str)
    cluster_parser.add_argument('experiment', type=str)
    cluster_parser.add_argument('-l', '--local', type=bool, default=False)


def _create_container_util_parser(subparsers) -> None:
    """
    Helper function to add container util execution arguments.
    @param subparsers: Subparser to add arguments to.
    @type subparsers: Any
    @return: None
    @rtype: None
    """
    util_docker_parser = subparsers.add_parser('util-docker')
    util_docker_parser.add_argument('name', type=str)
    util_docker_parser.add_argument('--clients', type=int)


def _create_util_parser(subparsers):
    """
    Helper function to add util generation execution arguments.
    @param subparsers: Subparser to add arguments to.
    @type subparsers: Any
    @return: None
    @rtype: None
    """
    util_generate_parser = subparsers.add_parser('util-generate')
    util_generate_parser.add_argument('path', type=str)


def _create_util_run_parser(subparsers) -> None:
    """
    Helper function to add util run execution arguments.
    @param subparsers: Subparser to add arguments to.
    @type subparsers: Any
    @return: None
    @rtype: None
    """
    util_run_parser = subparsers.add_parser('util-run')
    util_run_parser.add_argument('path', type=str)


def _create_remote_parser(subparsers) -> None:
    """
    Helper function to add remote Federated Learning execution arguments. Supports both Docker and K8s execution
    using optional (positional) arguments.
    @param subparsers: Subparser to add arguments to.
    @type subparsers: Any
    @return: None
    @rtype: None
    """
    remote_parser = subparsers.add_parser('remote')
    add_default_arguments(remote_parser)

    remote_parser.add_argument('rank', nargs='?', type=int, default=None)
    remote_parser.add_argument('--nic', type=str, default=None)
    remote_parser.add_argument('--host', type=str, default=None)


def _create_single_parser(subparsers) -> None:
    """
    Helper function to add Local single machine execution arguments.
    @param subparsers: Subparser to add arguments to.
    @type subparsers: Any
    @return: None
    @rtype: None
    """
    single_machine_parser = subparsers.add_parser('single')
    add_default_arguments(single_machine_parser)


def add_default_arguments(*parsers):
    """
    Helper function to add default arguments shared between executions.
    @param parsers: Subparser to add arguments to.
    @type subparsers: Any
    @return: None
    @rtype: None
    """
    for parser in parsers:
        parser.add_argument('config', type=str, help='')
        parser.add_argument('--prefix', type=str, default=None)


def create_all_subparsers(subparsers: ArgumentParser):
    """
    Helper function to add all subparsers to an argparse object.
    @param subparsers: Subparser to add arguments to.
    @type subparsers: Any
    @return: None
    @rtype: ArgumentParser
    """
    _create_extractor_parser(subparsers)
    _create_client_parser(subparsers)
    _create_cluster_parser(subparsers)
    _create_container_util_parser(subparsers)
    _create_util_parser(subparsers)
    _create_util_run_parser(subparsers)
    _create_remote_parser(subparsers)
    _create_single_parser(subparsers)
