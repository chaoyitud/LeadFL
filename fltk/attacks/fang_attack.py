import torch
import numpy as np

from fltk.attacks import lie_nn_parameters
from fltk.attacks.my_lie_attack import lie_attack
from fltk.strategy import median, multiKrum, krum, fed_avg, bulyan
from fltk.strategy.aggregation.trmean import trmean


def fang_attack_on_one_layer(all_updates):
    """
    all_updates: the gradients/parameters of all chosen clients in one layer
    return: fang attacks gradient/parameter
    """
    all_updates = all_updates.type(torch.FloatTensor)
    if (len(all_updates.shape) == 1):
        all_updates = all_updates.reshape(-1, 1)
    grad_shape = all_updates[0].shape
    model_re = torch.mean(all_updates, 0)
    model_std = torch.std(all_updates, 0)
    deviation = torch.sign(model_re)

    max_vector_low = model_re + 3 * model_std
    max_vector_hig = model_re + 4 * model_std
    min_vector_low = model_re - 4 * model_std
    min_vector_hig = model_re - 3 * model_std

    max_range = torch.cat((max_vector_low[:, None], max_vector_hig[:, None]), dim=1)
    min_range = torch.cat((min_vector_low[:, None], min_vector_hig[:, None]), dim=1)

    rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), 1])).type(torch.FloatTensor)

    max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack(
        [max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
    min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack(
        [min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T

    max_rand = max_rand.reshape(grad_shape)
    min_rand = min_rand.reshape(grad_shape)

    deviation_dim = (deviation > 0).float() * max_rand + (deviation < 0).float() * min_rand
    random_12 = 1. + torch.rand(size=grad_shape)
    return deviation_dim * ((deviation * deviation_dim > 0).float() / random_12 + (
                deviation * deviation_dim < 0).float() * random_12)


def fang_nn_parameters(dict_parameters):
    """
    generate fang parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """

    mal_param = {}
    for name in dict_parameters[list(dict_parameters.keys())[0]].keys():
        all_updates = []

        for idx in dict_parameters.keys():
            all_updates.append(dict_parameters[idx][name])

        mal_param[name] = fang_attack_on_one_layer(torch.stack(all_updates))
    return mal_param


if __name__ == '__main__':
    # test lie_attack
    benign_client_gradients = {'client_1': {'layer_1': torch.tensor([[1., 2., 3.], [4., 5., 6.]])+1,
                                            'layer_2': torch.tensor([[7., 8., 9.], [10., 11., 12.]])+5},
                               'client_2': {'layer_1': torch.tensor([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])+10,
                                            'layer_2': torch.tensor([[7.7, 8.8, 9.9], [10.1, 11.1, 12.12]])-10},
                               'client_3': {'layer_1': torch.tensor([[1.3, 2.4, 3.2], [4.5, 5.0, 6.9]])-10,
                                            'layer_2': torch.tensor([[7., 8., 9.], [10.1, 11.9, 12.9]])-0.2+10},
                                 'client_4': {'layer_1': torch.tensor([[1., 2., 3.], [4., 5., 6.]])-0.3,
                                            'layer_2': torch.tensor([[7., 8., 9.], [10., 11., 12.]])+0.4}}
    # add random int to all gradients
    for idx in benign_client_gradients.keys():
        for layer in benign_client_gradients[idx].keys():
            benign_client_gradients[idx][layer] += torch.randint(0, 10, benign_client_gradients[idx][layer].shape)
    print(benign_client_gradients)
    mal_weights = fang_nn_parameters(benign_client_gradients)
    benign_client_gradients['client_10'] = mal_weights
    benign_client_gradients['client_11'] = mal_weights
    benign_client_gradients['client_12'] = mal_weights
    benign_client_gradients['client_13'] = mal_weights
    print(benign_client_gradients)
    print(bulyan(benign_client_gradients, None))
    print(mal_weights)
