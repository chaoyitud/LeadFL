import copy
from typing import Tuple, Any

import numpy as np
import sklearn
import time
import torch
import yaml

from fltk.core.node import Node
from fltk.schedulers import MinCapableStepLR
from fltk.strategy import get_optimizer
from fltk.util.config import Config
from backdoors.helper import Helper
class Client(Node):
    """
    Federated experiment client.
    """
    running = False

    def __init__(self, identifier: str, rank: int, world_size: int, config: Config, mal: bool = False,
                 mal_loader: Any = None, backdoor_helper: Helper = None):
        super().__init__(identifier, rank, world_size, config)

        self.loss_function = self.config.get_loss_function()()
        self.optimizer = get_optimizer(self.config.optimizer)(self.net.parameters(),
                                                              **self.config.optimizer_args)
        self.scheduler = MinCapableStepLR(self.optimizer,
                                          self.config.scheduler_step_size,
                                          self.config.scheduler_gamma,
                                          self.config.min_lr)
        self.defense = self.config.defense
        self.mal = mal
        self.mal_loader = mal_loader if mal else None
        self.hessian_metrix = []
        self.regular_loss = 0.0
        self.regular_schedule = 1.0
        self.backdoor_helper = backdoor_helper

    def remote_registration(self):
        """
        Function to perform registration to the remote. Currently, this will connect to the Federator Client. Future
        version can provide functionality to register to an arbitrary Node, including other Clients.
        @return: None.
        @rtype: None
        """
        self.logger.info('Sending registration')
        self.message('federator', 'ping', 'new_sender')
        self.message('federator', 'register_client', self.id, self.rank)
        self.running = True
        self._event_loop()

    def stop_client(self):
        """
        Function to stop client after training. This allows remote clients to stop the client within a specific
        timeframe.
        @return: None
        @rtype: None
        """
        self.logger.info('Got call to stop event loop')
        self.running = False

    def _event_loop(self):
        self.logger.info('Starting event loop')
        while self.running:
            time.sleep(0.1)
        self.logger.info('Exiting node')

    def train(self, num_epochs: int, start_defense=True):
        """
        Function implementing federated learning training loop.
        @param num_epochs: Number of epochs to run.
        @type num_epochs: int
        @return: Final running loss statistic and acquired parameters of the locally trained network.
        @rtype: Tuple[float, Dict[str, torch.Tensor]]

        Args:
            start_defense: A flag to start defense
        """
        start_time = time.time()

        running_loss = 0.0
        final_running_loss = 0.0
        regular_loss = 0.0
        final_regular_loss = 0.0

        if self.distributed:
            self.dataset.train_sampler.set_epoch(num_epochs)

        if self.mal and self.config.attack_client is not None:
            if self.config.attack_client == 'TargetedAttack':
                number_of_training_samples = len(self.mal_loader)

                self.logger.info(f'{self.id}: Number of training samples: {number_of_training_samples}')

                for epoch in range(self.config.attack_epochs):
                    for i, (inputs, labels, _) in enumerate(self.mal_loader):
                        print(inputs.shape)
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        outputs = self.net(inputs)
                        loss = self.loss_function(outputs, labels)
                        loss.backward()
                        self.optimizer.step()
                        running_loss += loss.item()
                        if i % self.config.log_interval == 0:
                            self.logger.info(
                                f'[{self.id}] [{epoch:d}, {i:5d}] loss: {running_loss / self.config.log_interval:.3f}')
                            final_running_loss = running_loss / self.config.log_interval
                            running_loss = 0.0
                            # break

            elif self.config.attack_client == 'LabelFlip':
                number_of_training_samples = len(self.dataset.get_train_loader())
                self.logger.info(f'{self.id}: Number of training samples: {number_of_training_samples}')
                for epoch in range(num_epochs):
                    for i, (inputs, labels) in enumerate(self.dataset.get_train_loader(), 0):
                        # random choose 1 to 9
                        random_number = np.random.randint(1, 10)
                        labels = (labels + random_number) % 10
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        self.optimizer.zero_grad()
                        outputs = self.net(inputs)
                        loss = self.loss_function(outputs, labels)
                        loss.backward()
                        self.optimizer.step()
                        running_loss += loss.item()
                        if i % self.config.log_interval == 0:
                            self.logger.info(
                                f'[{self.id}] [{epoch:d}, {i:5d}] loss: {running_loss / self.config.log_interval:.3f}')
                            final_running_loss = running_loss / self.config.log_interval
                            running_loss = 0.0
                            # break
            elif self.config.attack_client == 'Backdoor':
                number_of_training_samples = len(self.dataset.get_train_loader())
                self.logger.info(f'{self.id}: Number of training samples: {number_of_training_samples}')
                for epoch in range(num_epochs):
                    for i, (inputs, labels) in enumerate(self.dataset.get_train_loader(), 0):
                        batch = self.backdoor_helper.task.get_batch(i, (inputs, labels))
                        self.optimizer.zero_grad()
                        loss = self.backdoor_helper.attack.compute_blind_loss(self.net, self.loss_function, batch, True)
                        loss.backward()
                        self.optimizer.step()
                        running_loss += loss.item()
                        if i % self.config.log_interval == 0:
                            self.logger.info(
                                f'[{self.id}] [{epoch:d}, {i:5d}] loss: {running_loss / self.config.log_interval:.3f}')
                            final_running_loss = running_loss / self.config.log_interval
                            running_loss = 0.0
                            # break

        else:
            number_of_training_samples = len(self.dataset.get_train_loader())
            self.logger.info(f'{self.id}: Number of training samples: {number_of_training_samples}')
            for epoch in range(num_epochs):
                old_gradient = {}
                old_gradient_mine = {}
                old_params = {}
                for i, (inputs, labels) in enumerate(self.dataset.get_train_loader(), 0):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.loss_function(outputs, labels)
                    running_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()

                    if not self.config.defense_half:
                        defense_half = True
                    elif epoch < num_epochs // 2:
                        defense_half = True
                    else:
                        defense_half = False

                    if self.config.defense in ["myWBC", "LeadFL", "LeadFLnorm"] and defense_half:
                        if i != 0:
                            loss = None
                            for name, param in self.net.named_parameters():
                                if 'weight' in name:
                                    ones = torch.ones(param.shape).to(self.device)
                                    regularization = torch.norm(
                                        param-old_params[name].detach().to(self.device) - old_gradient_mine[name].detach().to(self.device) - ones * self.config.lr)
                                    loss = loss + regularization if loss else regularization
                            loss = self.config.regular_weight * self.regular_schedule * loss
                            regular_loss += loss.item()
                            loss.backward()
                            if self.config.defense == "LeadFL":
                                torch.nn.utils.clip_grad_value_(self.net.parameters(), self.config.pert_strength)
                            # get number of nn parameters in the model
                            if self.config.defense == "LeadFLnorm":
                                num_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
                                pertubation = np.random.laplace(0, self.config.pert_strength, size=num_params)
                                # calculate l2 norm of the pertubation
                                l2_norm = np.linalg.norm(pertubation)
                                torch.nn.utils.clip_grad_norm_(self.net.parameters(), l2_norm)
                            self.optimizer.step()
                        for name, param in self.net.named_parameters():
                            if 'weight' in name:
                                old_gradient_mine[name] = copy.deepcopy(param.grad)
                                old_params[name] = copy.deepcopy(param)

                    if i != 0:
                        changed_ele_num = 0
                        all_ele_num = 0
                        changed_magnitude = 0
                        for name, p in self.net.named_parameters():
                            if 'weight' in name:
                                grad_tensor = p.grad.data.cpu().numpy()
                                grad_diff = grad_tensor - old_gradient[name]
                                pertubation = np.random.laplace(0, self.config.pert_strength,
                                                                size=grad_tensor.shape).astype(
                                    np.float32)

                                # calculate the change percentage of the gradient
                                changed_ele_num += np.sum(np.abs(grad_diff) > np.abs(pertubation))
                                all_ele_num += grad_diff.size
                                changed_magnitude += np.sum(np.abs(grad_diff))

                                if self.defense == "WBC":
                                    pertubation = np.where(abs(grad_diff) > abs(pertubation), 0, pertubation)
                                if (self.defense == "WBC" or self.defense == "LDP") and start_defense:
                                    p.data = torch.from_numpy(p.data.cpu().numpy() + pertubation * self.config.lr).to(
                                        self.device)

                        hessian_matrix_dict = {'ChangedPercent': changed_ele_num / all_ele_num,
                                               'ChangedMagnitude': changed_magnitude}
                        self.hessian_metrix.append(hessian_matrix_dict)

                    for name, p in self.net.named_parameters():
                        if 'weight' in name:
                            old_gradient[name] = copy.deepcopy(p.grad.data.cpu().numpy())
                    # Mark logging update step
                    if i % self.config.log_interval == 0:
                        self.logger.info(
                            f'[{self.id}] [{epoch:d}, {i:5d}] loss: {running_loss / self.config.log_interval:.3f}')
                        final_running_loss = running_loss / self.config.log_interval
                        final_regular_loss = regular_loss / self.config.log_interval
                        running_loss = 0.0
                        regular_loss = 0.0
                        # break
        self.regular_loss = final_regular_loss
        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f'Train duration is {duration} seconds')

        return final_running_loss, self.get_nn_parameters()

    def set_tau_eff(self, total):
        client_weight = self.get_client_datasize() / total
        n = self.get_client_datasize()  # pylint: disable=invalid-name
        E = self.config.epochs  # pylint: disable=invalid-name
        B = 16  # nicely hardcoded :) # pylint: disable=invalid-name
        tau_eff = int(E * n / B) * client_weight
        if hasattr(self.optimizer, 'set_tau_eff'):
            self.optimizer.set_tau_eff(tau_eff)

    def test(self) -> Tuple[float, float, np.array]:
        """
        Function implementing federated learning test loop.
        @return: Statistics on test-set given a (partially) trained model; accuracy, loss, and confusion matrix.
        @rtype: Tuple[float, float, np.array]
        """
        start_time = time.time()
        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0
        with torch.no_grad():
            for (images, labels) in self.dataset.get_test_loader():
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.net(images)

                _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += self.loss_function(outputs, labels).item()

        # Calculate learning statistics
        loss /= len(self.dataset.get_test_loader().dataset)
        accuracy = 100.0 * correct / total

        confusion_mat = sklearn.metrics.confusion_matrix(targets_, pred_)

        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f'Test duration is {duration} seconds')
        return accuracy, loss, confusion_mat

    def mal_test(self) -> Tuple[float, float, np.array]:
        start_time = time.time()
        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0
        with torch.no_grad():
            for (images, labels, _) in self.mal_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.net(images)

                _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += self.loss_function(outputs, labels).item()

        # Calculate learning statistics
        loss /= len(self.dataset.get_test_loader().dataset)
        accuracy = 100.0 * correct / total

        confusion_mat = sklearn.metrics.confusion_matrix(targets_, pred_)

        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f'Test duration is {duration} seconds')
        return accuracy, loss, confusion_mat

    def backdoor_test(self) -> Tuple[float, float, np.array]:
        """
        Function implementing federated learning test loop.
        @return: Statistics on test-set given a (partially) trained model; accuracy, loss, and confusion matrix.
        @rtype: Tuple[float, float, np.array]
        """
        start_time = time.time()
        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0
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

        # Calculate learning statistics
        loss /= len(self.dataset.get_test_loader().dataset)
        accuracy = 100.0 * correct / total

        confusion_mat = sklearn.metrics.confusion_matrix(targets_, pred_)

        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f'Test duration is {duration} seconds')
        return accuracy, loss, confusion_mat

    def get_client_datasize(self):  # pylint: disable=missing-function-docstring
        return len(self.dataset.get_train_sampler())

    def get_client_status(self):
        return self.mal

    def get_client_hessian(self):
        return self.hessian_metrix

    def get_client_regular_loss(self):
        return self.regular_loss

    def set_client_regular_schedule(self, communication_round, decay_rate=0.8):
        self.regular_schedule = decay_rate**(int(communication_round/100))

    def exec_round(self, num_epochs: int, start_defense=True) -> Tuple[
        Any, Any, Any, Any, float, float, float, np.array]:
        """
        Function as access point for the Federator Node to kick off a remote learning round on a client.
        @param num_epochs: Number of epochs to run
        @type num_epochs: int
        @return: Tuple containing the statistics of the training round; loss, weights, accuracy, test_loss, make-span,
        training make-span, testing make-span, and confusion matrix.
        @rtype: Tuple[Any, Any, Any, Any, float, float, float, np.array]
        """
        start = time.time()
        loss, weights = self.train(num_epochs, start_defense)
        time_mark_between = time.time()
        test_conf_matrix = None
        if self.mal:
            if self.config.attack_client == "Targeted":
                accuracy, test_loss, confusion_mat = self.mal_test()
            elif self.config.attack_client == "Backdoor":
                accuracy, test_loss, confusion_mat = self.backdoor_test()
            else:
                accuracy, test_loss, test_conf_matrix = self.test()
        else:
            accuracy, test_loss, test_conf_matrix = self.test()

        end = time.time()
        round_duration = end - start
        train_duration = time_mark_between - start
        test_duration = end - time_mark_between
        # self.logger.info(f'Round duration is {duration} seconds')

        if hasattr(self.optimizer, 'pre_communicate'):  # aka fednova or fedprox
            self.optimizer.pre_communicate()
        for k, value in weights.items():
            weights[k] = value.cpu()
        return loss, weights, accuracy, test_loss, round_duration, train_duration, test_duration, test_conf_matrix

    def __del__(self):
        self.logger.info(f'Client {self.id} is stopping')


if __name__ == '__main__':
    num_params = 100
    pertubation = np.random.laplace(0, 1, size=num_params)
    # calculate l2 norm of the pertubation
    l2_norm = np.linalg.norm(pertubation)
    print(pertubation)
    print(l2_norm)
