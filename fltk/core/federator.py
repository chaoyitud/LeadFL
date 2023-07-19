import copy
import re
from turtle import clear

import numpy as np
import sklearn
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Tuple
import torch.nn.functional as F
import torch
import os


import wandb

from fltk.core.client import Client
from fltk.core.node import Node
from fltk.datasets.loader_util import get_dataset
from fltk.strategy import get_aggregation
from fltk.strategy import random_selection
from fltk.util.config import Config
from fltk.util.config.definitions import Nets
from fltk.util.data_container import DataContainer, FederatorRecord, ClientRecord
from fltk.attacks import get_attack
from backdoors.helper import Helper
from fltk.strategy.aggregation.clustering import clustering
import yaml
from fltk.strategy.aggregation.clustering import calculate_cluster_metrics

NodeReference = Union[Node, str]


@dataclass
class LocalClient:
    """
    Dataclass for local execution references to 'virtual' Clients.
    """
    name: str
    ref: NodeReference
    data_size: int
    exp_data: DataContainer


def cb_factory(future: torch.Future, method, *args, **kwargs):  # pylint: disable=no-member
    """
    Callback factory function to attach callbacks to a future.
    @param future: Future promise for remote function.
    @type future: torch.Future.
    @param method: Callable method to call on a remote.
    @type method: Callable
    @param args: Arguments to pass to the callback function.
    @type args: List[Any]
    @param kwargs: Keyword arguments to pass to the callback function.
    @type kwargs: Dict[str, Any]
    @return: None
    @rtype: None
    """
    future.then(lambda x: method(x, *args, **kwargs))


class Federator(Node):
    """
    Federator implementation that governs the (possibly) distributed learning process. Learning is initiated by the
    Federator and performed by the Clients. The Federator also performs centralized logging for easier execution.
    """
    clients: List[LocalClient] = []
    benign_clients: List[LocalClient] = []
    mal_clients: List[LocalClient] = []

    # clients: List[NodeReference] = []
    num_rounds: int
    exp_data: DataContainer

    def __init__(self, identifier: str, rank: int, world_size: int, config: Config):
        super().__init__(identifier, rank, world_size, config)
        self.cluster_acc = 0.0
        self.cluster_precision = 0.0
        self.cluster_recall = 0.0
        self.loss_function = self.config.get_loss_function()()
        self.num_rounds = config.rounds
        self.config = config
        prefix_text = ''
        if config.replication_id:
            prefix_text = f'_r{config.replication_id}'
        config.output_path = Path(config.output_path) / f'{config.experiment_prefix}{prefix_text}'
        self.exp_data = DataContainer('federator', config.output_path, FederatorRecord, config.save_data_append)
        self.aggregation_method = get_aggregation(config.aggregation)
        self.backdoor_acc_list = []
        self.mal_confidence_list= []
        self.mal_loader = None
        self.defense_controller_counter = 0
        self.true_mal = {}
        self.stored_gradients = []
        self.stored_index = []
        self.acc_list = []
        if self.config.attack_client == 'Backdoor':
            with open(self.config.backdoor_config, 'r') as file:
                backdoor_params = yaml.load(file, Loader=yaml.FullLoader)
                if self.config.backdoor_type == '1-pixel':
                    backdoor_params['synthesizer'] = 'SinglePixel'
                elif self.config.backdoor_type == '9-pixel':
                    backdoor_params['synthesizer'] = 'Pattern'
            self.backdoor_helper = Helper(backdoor_params)
        else:
            self.backdoor_helper = None

    def create_clients(self):
        """
        Function to create references to all the clients that will perform the learning process.
        @return: None.
        @rtype: None
        """
        self.logger.info('Creating clients')
        if self.config.single_machine:
            # Create direct clients
            world_size = self.config.num_clients + 1
            client_list = [i for i in range(1, self.config.world_size)]
            np.random.seed(0)
            mal_list = np.random.choice(client_list, self.config.num_mal_clients, replace=False)
            self.true_mal = mal_list
            self.logger.info(f'These Malicious clients are selected: {mal_list}')
            for client_id in range(1, self.config.world_size):
                client_name = f'client{client_id}'
                mal = True if client_id in mal_list else False
                client = Client(client_name, client_id, world_size, copy.deepcopy(self.config), mal=mal,
                                mal_loader=self.mal_loader, backdoor_helper=self.backdoor_helper)
                self.clients.append(
                    LocalClient(client_name, client, 0, DataContainer(client_name, self.config.output_path,
                                                                      ClientRecord, self.config.save_data_append)))
                if mal:
                    self.mal_clients.append(self.clients[-1])
                else:
                    self.benign_clients.append(self.clients[-1])

                self.logger.info(f'Client "{client_name}" created')

    def register_client(self, client_name: str, rank: int):
        """
        Function to be called by remote Client to register the learner to the Federator.
        @param client_name: Name of the client.
        @type client_name: str
        @param rank: Rank of the client that registers.
        @type rank: int
        @return: None.
        @rtype: None
        """
        self.logger.info(f'Got new client registration from client {client_name}')
        if self.config.single_machine:
            self.logger.warning('This function should not be called when in single machine mode!')
        self.clients.append(
            LocalClient(client_name, client_name, rank, DataContainer(client_name, self.config.output_path,
                                                                      ClientRecord, self.config.save_data_append)))

    def stop_all_clients(self):
        """
        Method to stop all running clients that were registered by the Federator during initialziation.
        @return: None.
        @rtype: None
        """
        for client in self.clients:
            self.message(client.ref, Client.stop_client)

    def _num_clients_online(self) -> int:
        return len(self.clients)

    def _all_clients_online(self) -> bool:
        return len(self.clients) == self.world_size - 1

    def clients_ready(self):
        """
        Synchronous implementation to wait for all remote Clients to register themselves to the Federator.
        """
        all_ready = False
        ready_clients = []
        while not all_ready:
            responses = []
            all_ready = True
            for client in self.clients:
                resp = self.message(client.ref, Client.is_ready)
                if resp:
                    self.logger.info(f'Client {client} is ready')
                else:
                    self.logger.info(f'Waiting for client {client}')
                    all_ready = False
            time.sleep(2)

    def get_client_data_sizes(self):
        """
        Function to request the dataset sizes of the Clients training DataLoaders.
        @return: None.
        @rtype: None
        """
        for client in self.clients:
            client.data_size = self.message(client.ref, Client.get_client_datasize)

    def run(self):
        """
        Spinner function to perform the experiment that was provided by either the Orhcestrator in Kubernetes or the
        generated docker-compose file in case running in Docker.
        @return: None.
        @rtype: None.
        """
        # Load dataset with world size 2 to load the whole dataset.
        # Caused by the fact that the dataloader subtracts 1 from the world size to exclude the federator by default.
        if self.config.use_wandb:
            self.init_wandb()
        self.init_dataloader(world_size=2)
        self.dataset.init_mal_dataset()
        self.mal_loader = self.dataset.get_mal_loaders()
        self.create_clients()
        while not self._all_clients_online():
            msg = f'Waiting for all clients to come online. ' \
                  f'Waiting for {self.world_size - 1 - self._num_clients_online()} clients'
            self.logger.info(msg)
            time.sleep(2)
        self.logger.info('All clients are online')
        # self.logger.info('Running')
        # time.sleep(10)
        self.client_load_data()
        self.get_client_data_sizes()
        self.clients_ready()
        # self.logger.info('Sleeping before starting communication')
        # time.sleep(20)
        for communication_round in range(self.config.rounds):
            self.exec_round(communication_round)

        self.save_data()
        self.logger.info('Federator is stopping')

    def save_data(self):
        """
        Function to store all data obtained from the Clients during their training efforts.
        @return: None.
        @rtype: None
        """
        self.exp_data.save()
        for client in self.clients:
            client.exp_data.save()

    def client_load_data(self):
        """
        Function to contact all clients to initialize their dataloaders, to be called to prepare the Clients for the
        learning loop.
        @return: None.
        @rtype: None
        """
        for client in self.clients:
            self.message(client.ref, Client.init_dataloader)

    def client_aggregate_hessian(self, selected_clients):
        """
        Function to contact all clients to aggregate their Hessian matrices.
        @return: None.
        @rtype: None
        """
        changed_percentage = []
        changed_magnitude = []
        for client in selected_clients:
            if not self.message(client.ref, Client.get_client_status):
                hessian_metrix = self.message(client.ref, Client.get_client_hessian)
                for h in hessian_metrix:
                    changed_percentage.append(h['ChangedPercent'])
                    changed_magnitude.append(h['ChangedMagnitude'])
        if len(changed_percentage) == 0:
            return 0, 0
        print(sum(changed_percentage) / len(changed_percentage), sum(changed_magnitude) / len(changed_magnitude))
        return sum(changed_percentage) / len(changed_percentage), sum(changed_magnitude) / len(changed_magnitude)

    def client_aggregate_regular_loss(self, selected_clients):
        """
        Function to aggregate regular loss
        Args:
            selected_clients:

        Returns:

        """
        final_regular_loss = 0.0
        num_benign = 0
        for client in selected_clients:
            if not self.message(client.ref, Client.get_client_status):
                regular_loss = self.message(client.ref, Client.get_client_regular_loss)
                final_regular_loss += regular_loss
                num_benign += 1
        if num_benign == 0:
            return 0
        return final_regular_loss / num_benign

    def set_tau_eff(self):
        total = sum(client.data_size for client in self.clients)
        # responses = []
        for client in self.clients:
            self.message(client.ref, Client.set_tau_eff, client.ref, total)
            # responses.append((client, _remote_method_async(Client.set_tau_eff, client.ref, total)))
        # torch.futures.wait_all([x[1] for x in responses])

    def set_regular_schedule(self, communication_round):
        for client in self.clients:
            self.message(client.ref, Client.set_client_regular_schedule, communication_round)

    def test(self, net) -> Tuple[float, float, np.array]:
        """
        Function to test the learned global model by the Federator. This does not take the client distributions in
        account but is centralized.
        @param net: Global network to be tested on the Federator centralized testing dataset.
        @type net: torch.nn.Module
        @return: Accuracy and loss of the global network on a (subset) of the testing data, and the confusion matrix
        corresponding to the models' predictions.
        @rtype: Tuple[float, float, np.array]
        """
        start_time = time.time()
        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0
        print(len(self.dataset.get_test_loader()))
        with torch.no_grad():
            for (images, labels) in self.dataset.get_test_loader():
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = net(images)

                _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += self.loss_function(outputs, labels).item()
        loss /= len(self.dataset.get_test_loader().dataset)
        accuracy = 100.0 * correct / total
        confusion_mat = sklearn.metrics.confusion_matrix(targets_, pred_)

        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f'Test duration is {duration} seconds')
        return accuracy, loss, confusion_mat

    def backdoor_test(self, net) -> Tuple[float, float, np.array]:
        """
        Function to test the learned global model by the Federator. This does not take the client distributions in
        account but is centralized.
        @param net: Global network to be tested on the Federator centralized testing dataset.
        @type net: torch.nn.Module
        @return: Accuracy and loss of the global network on a (subset) of the testing data, and the confusion matrix
        corresponding to the models' predictions.
        @rtype: Tuple[float, float, np.array]
        """
        start_time = time.time()
        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0
        confidence_sum = 0.0
        with torch.no_grad():
            for i, data in enumerate(self.dataset.get_test_loader()):
                batch = self.backdoor_helper.task.get_batch(i, data)
                batch = self.backdoor_helper.attack.synthesizer.make_backdoor_batch(batch, test=True, attack=True)
                outputs = self.net(batch.inputs)
                labels = batch.labels
                _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += self.loss_function(outputs, labels).item()
                confidence_sum += sum(F.softmax(outputs.data.detach(), dim=1).cpu().data[:, labels[0]])
        loss /= len(self.dataset.get_test_loader().dataset)
        accuracy = 100.0 * correct / total
        confidence = float(confidence_sum / total)

        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f'Backdoor Test duration is {duration} seconds')
        return accuracy, loss, confidence

    def mal_test(self, net) -> Tuple[float, float, float]:
        start_time = time.time()
        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0
        confidence_sum = 0.0
        with torch.no_grad():
            for (images, labels, labels_true) in self.mal_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = net(images)

                _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += self.loss_function(outputs, labels).item()
                label_list = []
                idx_list = []
                for i in range(len(labels)):
                    idx_list.append(int(i))
                    label_list.append([int(labels[i].item())])
                if self.config.net_name == Nets.cifar10_cnn:
                    confidence_sum += sum(outputs.data.detach().cpu().data[idx_list, label_list])
                else:
                    confidence_sum += sum(F.softmax(outputs.data.detach(), dim=1).cpu().data[idx_list, label_list])

        loss /= len(self.dataset.get_test_loader().dataset)
        accuracy = 100.0 * correct / total
        confidence = float(confidence_sum / total)

        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f'Backdoor Test duration is {duration} seconds')
        return accuracy, loss, confidence

    def exec_round(self, com_round_id: int):
        """
        Helper method to call a remote Client to perform a training round during the training loop.
        @param com_round_id: Identifier of the communication round.
        @type com_round_id: int
        @return: None
        @rtype: None
        """
        start_time = time.time()
        num_epochs = self.config.epochs

        # Client selection
        selected_clients: List[LocalClient]
        np.random.seed(com_round_id + 100)  # 100 200 300
        mal_this_round = 0

        if self.config.selection_method == 'random':
            selected_clients = random_selection(self.clients, self.config.clients_per_round)

        elif self.config.selection_method == 'consistent':
            selected_benign_clients = random_selection(self.benign_clients,
                                                       self.config.clients_per_round - self.config.mal_clients_per_round)
            selected_mal_clients = random_selection(self.mal_clients, self.config.mal_clients_per_round)
            # combine benign and malicious clients
            selected_clients = np.concatenate((selected_benign_clients, selected_mal_clients))
        elif self.config.selection_method == 'periodic':
            if com_round_id % self.config.periodic_selection_period in [0, 1, 2, 3, 4]:
                selected_benign_clients = random_selection(self.benign_clients,
                                                           self.config.clients_per_round - self.config.mal_clients_per_round)
                selected_mal_clients = random_selection(self.mal_clients, self.config.mal_clients_per_round)
                selected_clients = np.concatenate((selected_benign_clients, selected_mal_clients))
            else:
                selected_clients = random_selection(self.benign_clients, self.config.clients_per_round)
        elif self.config.selection_method == 'random_clients':
            # random choose number from  [0, 0, 2, 2, 5]
            mal_this_round = np.random.choice([0, 0, 2, 2, 5])
            selected_benign_clients = random_selection(self.benign_clients,
                                                       self.config.clients_per_round - mal_this_round)
            selected_mal_clients = random_selection(self.mal_clients, mal_this_round)
            selected_clients = np.concatenate((selected_benign_clients, selected_mal_clients))

        if com_round_id in [0, 10, 12]:
            selected_benign_clients = random_selection(self.benign_clients, 5)
            selected_mal_clients = random_selection(self.mal_clients, 5)
            # combine benign and malicious clients
            selected_clients = np.concatenate((selected_benign_clients, selected_mal_clients))

        # Regular schedule to avoid the regular loss to be too high
        if self.config.regular_schedule:
            if com_round_id % 100 == 0:
                self.set_regular_schedule(com_round_id)

        mal_this_round = 0
        # Malicious selection
        for client in selected_clients:
            if self.message(client.ref, Client.get_client_status):
                self.logger.info(f'Malicious client {client.ref} is selected')
                mal_this_round += 1
        self.logger.info(f'This round {mal_this_round} malicious clients are selected')
        clients_status = [self.message(client.ref, Client.get_client_status) for client in selected_clients]
        last_model = self.get_nn_parameters()

        for client in selected_clients:
            self.message(client.ref, Client.update_nn_parameters, last_model)
        # Actual training calls
        client_weights = {}
        client_sizes = {}
        # pbar = tqdm(selected_clients)
        # for client in pbar:

        # Client training
        training_futures: List[torch.Future] = []  # pylint: disable=no-member

        def training_cb(fut: torch.Future, client_ref: LocalClient, client_weights, client_sizes,
                        num_epochs):  # pylint: disable=no-member
            train_loss, weights, accuracy, test_loss, round_duration, train_duration, test_duration, c_mat = fut.wait()
            self.logger.info(f'Training callback for client {client_ref.name} with accuracy={accuracy}')
            client_weights[client_ref.name] = weights
            client_data_size = self.message(client_ref.ref, Client.get_client_datasize)
            client_sizes[client_ref.name] = client_data_size
            c_record = ClientRecord(com_round_id, train_duration, test_duration, round_duration, num_epochs, 0,
                                    accuracy, train_loss, test_loss, confusion_matrix=c_mat)
            client_ref.exp_data.append(c_record)

        # defense controller
        start_defense = True
        if self.config.defense_controller:
            if self.defense_controller_counter > 0:
                start_defense = True
                self.defense_controller_counter -= 1
            else:
                start_defense = False

        for client in selected_clients:
            future = self.message_async(client.ref, Client.exec_round, num_epochs, start_defense)
            cb_factory(future, training_cb, client, client_weights, client_sizes, num_epochs)
            self.logger.info(f'Request sent to client {client.name}')
            training_futures.append(future)

        def all_futures_done(futures: List[torch.Future]) -> bool:  # pylint: disable=no-member
            return all(map(lambda x: x.done(), futures))

        while not all_futures_done(training_futures):
            time.sleep(0.1)
            self.logger.info('')
            # self.logger.info(f'Waiting for other clients')

        self.logger.info('Continue with rest [1]')
        time.sleep(0.5)
        mal_boost = self.config.mal_boost

        mal_name = []
        for client in selected_clients:
            if self.message(client.ref, Client.get_client_status):
                mal_name.append(client.name)

        if self.config.attack_server is not None and mal_this_round > 0:
            mal_weights = get_attack(self.config.attack_server)(client_weights)
            client_weights[mal_name[0]] = mal_weights
            for mal_nam in mal_name:
                client_weights[mal_nam] = mal_weights
        boost_cnt = 0
        if self.config.mal_boost > 1 and mal_this_round > 0:
            # assert then print error message
            assert self.config.mal_boost >= mal_this_round, 'mal_boost is less than mal_this_round'

            for name in mal_name[1:]:
                if mal_boost > 1:
                    client_weights[name] = copy.deepcopy(client_weights[mal_name[0]])
                    client_sizes[name] = client_sizes[mal_name[0]]
                    mal_boost -= 1
                    boost_cnt += 1

            for client in selected_clients:
                if mal_boost > 1 and not self.message(client.ref, Client.get_client_status):
                    client_weights[client.name] = copy.deepcopy(client_weights[mal_name[0]])
                    client_sizes[client.name] = client_sizes[mal_name[0]]
                    mal_boost -= 1
                    boost_cnt += 1

        print(f'Boosted {boost_cnt} clients')

        # calculate the gradients for clustering
        if self.config.cluster is True:
            client_gradients = self.calculate_gradients(client_weights)
            # print(f'Client gradients: {client_gradients}')
            # update the gradients
            self.update_stored_gradients(client_gradients)
            majority_id, candidates = clustering(self.stored_gradients, self.stored_index,
                                                 self.config.clients_per_round, self.config.cluster_stored_rounds)
            print(f'Majority id: {majority_id}')
            print(f'Candidate ids: {candidates}')
            # only selected candidates clients
            if len(candidates) > 0:
                client_weights = {k: v for k, v in client_weights.items() if k in candidates}
                client_sizes = {k: v for k, v in client_sizes.items() if k in candidates}
                # num of candidates in mal_name
                self.cluster_acc, self.cluster_recall, self.cluster_precision = calculate_cluster_metrics(
                    self.stored_index[-self.config.clients_per_round:],
                    mal_name, candidates)

        updated_model = self.aggregation_method(client_weights,
                                                client_sizes) if self.config.aggregation != 'trmean' \
            else self.aggregation_method(client_weights, client_sizes, self.config.tm_beta)
        if self.config.cluster is False or len(candidates) > 0:
            self.update_nn_parameters(updated_model)
        test_accuracy, test_loss, conf_mat = self.test(self.net)
        mal_accuracy, mal_loss, mal_confidence = self.mal_test(self.net)
        if self.config.attack_client == 'Backdoor':
            backdoor_accuracy, backdoor_loss, backdoor_confidence = self.backdoor_test(self.net)
        else:
            backdoor_accuracy = 0
            backdoor_loss = 0
            backdoor_confidence = 0
        self.logger.info(f'[Round {com_round_id:>3}] Federator has a accuracy of {test_accuracy} and loss={test_loss}')
        self.logger.info(
            f'[Round {com_round_id:>3}] Federator has a Malicious accuracy of {mal_accuracy} and loss={mal_loss}, malicious confidence={mal_confidence}')
        end_time = time.time()
        duration = end_time - start_time
        record = FederatorRecord(len(selected_clients), com_round_id, duration, test_loss, test_accuracy, mal_loss,
                                 mal_accuracy, mal_confidence,
                                 confusion_matrix=conf_mat)

        changed_percentage, changed_magnitude = self.client_aggregate_hessian(selected_clients)
        regular_loss = self.client_aggregate_regular_loss(selected_clients)

        # keep the backdoor average accuracy of every ten rounds
        if self.config.attack_client == 'Backdoor':
            self.backdoor_acc_list.append(backdoor_accuracy)
            if com_round_id == self.config.rounds - 1:
                self.backdoor_acc_list = np.array(self.backdoor_acc_list)
                backdoor_acc_list = []
                post_round = 10 if self.config.rounds > 10 else self.config.rounds
                for i in range(post_round):
                    backdoor_acc_list.append(np.mean(self.backdoor_acc_list[i::10]))
                    wandb.summary[f'backdoor_acc{i}'] = backdoor_acc_list[-1]
        self.mal_confidence_list.append(float(mal_confidence))
        self.acc_list.append(float(test_accuracy))
        if com_round_id == self.config.rounds - 1:
            wandb.summary['global_acc'] = test_accuracy
            wandb.summary['attack'] = self.config.attack_client
            if self.config.attack_client == 'Backdoor':
                wandb.summary['backdoor_type'] = self.config.backdoor_type
            wandb.summary['aggregation'] = self.config.aggregation.value
            wandb.summary['defense'] = self.config.defense
            sum_table = wandb.Table(
                columns=['TargetedAttack', 'UntargetedAttack', 'defense', 'aggregation', 'backdoor_type', 'clustering',
                         'global_acc', 'backdoor_accuracy_avg',
                         'backdoor_accuracy_final', 'malicious_confidence_avg', 'malicious_confidence_final', 'global_acc_highest'])
            sum_table.add_data(self.config.attack_client,
                               self.config.attack_server.name if self.config.attack_server is not None else None,
                               self.config.defense,
                               self.config.aggregation.value,
                               self.config.backdoor_type if self.config.attack_client == 'Backdoor' else None,
                               self.config.cluster_stored_rounds if self.config.cluster else None,
                               test_accuracy,
                               np.mean(backdoor_acc_list) if self.config.attack_client == 'Backdoor' else None,
                               self.backdoor_acc_list[-1] if self.config.attack_client == 'Backdoor' else None,
                               np.mean(self.mal_confidence_list),
                               self.mal_confidence_list[-1],
                               np.max(self.acc_list))
            wandb.summary['summary'] = sum_table
            wandb.log({'summary': sum_table})
            artifact = wandb.Artifact('sum_table', 'dataset')
            artifact.add(sum_table, "sum_table")

            wandb.log_artifact(artifact)

        if self.config.use_wandb:
            wandb.log({"Cluster/accuracy": self.cluster_acc}, step=com_round_id)
            wandb.log({"Cluster/recall": self.cluster_recall}, step=com_round_id)
            wandb.log({"Cluster/precision": self.cluster_precision}, step=com_round_id)
            wandb.log({"Federator/Accuracy": test_accuracy, "Federator/Loss": test_loss,
                       "Federator/Malicious number this round": mal_this_round}, step=com_round_id)
            wandb.log({"Malicious/Accuracy": mal_accuracy, "Malicious/Loss": mal_loss,
                       "Malicious/Confidence": mal_confidence}, step=com_round_id)
            wandb.log({"round": com_round_id}, step=com_round_id)
            wandb.log(
                {"Federator/Changed percentage": changed_percentage, "Federator/Changed magnitude": changed_magnitude},
                step=com_round_id)
            wandb.log({"Federator/Regular loss": regular_loss}, step=com_round_id)
            wandb.log({"Backdoor/Backdoor accuracy": backdoor_accuracy, "Backdoor/Backdoor loss": backdoor_loss,
                       "Backdoor/Backdoor Confidence": backdoor_confidence}, step=com_round_id)
        self.exp_data.append(record)
        self.logger.info(f'[Round {com_round_id:>3}] Round duration is {duration} seconds')

        if self.config.defense_controller:
            self.defense_controller()

    def init_wandb(self):
        aggregation_name = self.config.aggregation.name
        if self.config.attack_server is not None:
            attack_name = self.config.attack_server.name
        elif self.config.attack_client is not None:
            attack_name = self.config.attack_client
        else:
            attack_name = 'noAttack'
        if attack_name == 'Backdoor':
            attack_name = 'Backdoor' + str(self.config.backdoor_type)
        defense_name = self.config.defense if self.config.defense is not None else 'noDefense'
        run_name = aggregation_name + '_' + attack_name + '_vs_' + defense_name
        if self.config.cluster == True:
            run_name = run_name + '_cluster' + str(self.config.cluster_stored_rounds)
        wandb.init(
            project=self.config.experiment_prefix + '_' + self.config.dataset_name.name + '_' + self.config.data_sampler.name,
            name=run_name, entity="tudlab")

        wandb.config = {
            "defense": self.config.defense,
            "dataset": self.config.dataset_name,
            "net_name": self.config.net_name,
            "data_sampler": self.config.data_sampler,
            "num_clients": self.config.num_clients,
            "pert_strength": self.config.pert_strength,
            "local_epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "lr": self.config.lr,
            "rounds": self.config.rounds,
            "mal_boost": self.config.mal_boost,
            "mal_samples": self.config.mal_samples,
            "num_mal_clients": self.config.num_mal_clients,
            "clients_per_round": self.config.clients_per_round
        }

    def defense_controller(self):
        """
        Defense controller
        This is a simple defense controller that start the defense when the accuracy decrease
        Returns:
        """
        if len(self.exp_data.records) > 1:
            if self.exp_data.records[-2].test_accuracy - self.exp_data.records[-1].test_accuracy > 0.5:
                self.logger.info("Accuracy Drop Detected, Start Defense")
                self.defense_controller_counter = 5 if self.defense_controller_counter == 0 else self.defense_controller_counter

    def calculate_gradients(self, client_weights: dict):
        """
        Calculate the gradients of the clients
        Args:
            client_weights: the weights of the clients
        Returns:
            gradients: the gradients of the clients
        """
        client_gradients = {}
        # get weights of net
        last_model = self.get_nn_parameters()
        # loop through all clients_weights
        for client_name in client_weights:
            gradients = {}
            # get client weights
            client_weights_dict = client_weights[client_name]
            # calculate the gradients
            for key in client_weights_dict:
                gradients[key] = client_weights_dict[key].data - last_model[key].data.to('cpu')
            client_gradients[client_name] = gradients
        return client_gradients

    def update_stored_gradients(self, clients_gradients):
        # init numpy array

        for client_name in clients_gradients:
            client_gradients = clients_gradients[client_name]
            # extract client number from the string like 'client1'
            # client_index = int(re.findall(r'\d+', client_name)[0])
            self.stored_index.append(client_name)
            gradients = [torch.flatten(client_gradients[key]) for key in client_gradients]
            gradients = torch.cat(gradients).view(1, -1)
            self.stored_gradients.append(gradients.numpy())


if __name__ == '__main__':
    def update_stored_gradients(clients_gradients):
        # init numpy array
        gradients_array = []
        clients_index = []
        for client_name in clients_gradients:
            client_gradients = clients_gradients[client_name]
            # extract client number from the string like 'client1'
            client_index = int(re.findall(r'\d+', client_name)[0])
            clients_index.append(client_index)
            gradients = [torch.flatten(client_gradients[key]) for key in client_gradients]
            gradients = torch.cat(gradients).view(1, -1)
            gradients_array.append(gradients.numpy())
        # convert to numpy array
        gradients_array = np.concatenate(gradients_array, axis=0)
        client_index = np.array(clients_index)
        return gradients_array, clients_index


    # test update_stored_gradients
    client_gradients = {'client1': {'conv1.weight': torch.tensor([[1, 2], [3, 4]]), 'conv1.bias': torch.tensor([1, 2])},
                        'client2': {'conv1.weight': torch.tensor([[1, 2], [3, 4]]), 'conv1.bias': torch.tensor([1, 2])}}
    a = update_stored_gradients(client_gradients)
    print(a[0].shape)
