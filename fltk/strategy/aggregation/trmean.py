# pylint: disable=invalid-name
from typing import Dict

import numpy as np
import torch

def trmean(g, f):
    """ Trimmed mean rule.
  Args:
    g Non-empty stack of gradients to aggregate
    f Number of Byzantine gradients to tolerate
  Returns:
    Aggregated gradient
  """
    # Compute average of "inner" values
    if f == 0:
        return g.mean(dim=0)
    return g.sort(dim=0).values[f:-f].mean(dim=0)


def trimmed_mean(parameters: Dict[str, Dict[str, torch.Tensor]], sizes: Dict[str, int], beta=0.4) -> Dict[
    str, torch.Tensor]:
    """
    Function to perform FederatedAveraging with on a list of parameters.
    @param parameters: Dictionary of per-client provided Parameters.
    @type parameters:  Dict[str, Dict[str, torch.Tensor]]
    @param sizes: Dictionary of size descriptions for volume of data on client (to weight accordingly).
    @type sizes: Dict[str, int]
    @return: New parameters for next round.
    @rtype: Dict[str, torch.Tensor]
    """
    new_params = {}
    sum_size = 0
    # For each client
    for client in parameters:
        # For each module in the client
        for name in parameters[client].keys():
            try:
                new_params[name].append(parameters[client][name].data.float())
            except:
                new_params[name] = []
                new_params[name].append(parameters[client][name].data.float())

    for name in new_params:
        new_params[name] = trmean(torch.stack(new_params[name]), int(beta * len(new_params[name])))

    return new_params


if __name__ == "__main__":
    # test the median function
    params = {'1': {'1': torch.Tensor([[1, 2, 3], [4, 5, 6]]), '2': torch.Tensor([[7, 8, 9], [10, 11, 12]])},
              '2': {'1': torch.Tensor([[13, 14, 15], [16, 17, 18]]), '2': torch.Tensor([[19, 20, 21], [22, 23, 24]])},
              '3': {'1': torch.Tensor([[25, 26, 27], [28, 29, 30]]), '2': torch.Tensor([[31, 32, 33], [34, 35, 36]])},
              '4': {'1': torch.Tensor([[37, 38, 39], [40, 41, 42]]), '2': torch.Tensor([[43, 44, 45], [46, 47, 48]])},
              '5': {'1': torch.Tensor([[49, 50, 51], [52, 53, 54]]), '2': torch.Tensor([[55, 56, 57], [58, 59, 60]])},
              '6': {'1': torch.Tensor([[61, 62, 63], [64, 65, 66]]), '2': torch.Tensor([[67, 68, 69], [70, 71, 72]])},
              '7': {'1': torch.Tensor([[73, 74, 75], [76, 77, 78]]), '2': torch.Tensor([[79, 80, 81], [82, 83, 84]])},
              '8': {'1': torch.Tensor([[85, 86, 87], [88, 89, 90]]), '2': torch.Tensor([[91, 92, 93], [94, 95, 96]])},
              '9': {'1': torch.Tensor([[97, 98, 99], [100, 101, 102]]), '2': torch.Tensor([[103, 104, 105], [106, 107, 108]])},
              '10': {'1': torch.Tensor([[109, 110, 111], [112, 113, 114]]), '2': torch.Tensor([[115, 116, 117], [118, 119, 120]])}}
    sizes = {'1': 2, '2': 2, '3': 3}
    print(trimmed_mean(params, sizes, 0.4))
    # test torch stack
    # long tensor to float tensor
    print(torch.stack([torch.Tensor([1, 2, 3]), torch.Tensor([4, 5, 6])]).float())

