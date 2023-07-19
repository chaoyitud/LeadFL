# pylint: disable=invalid-name
from typing import Dict

import numpy as np
import torch


def median(parameters: Dict[str, Dict[str, torch.Tensor]], sizes: Dict[str, int]) -> Dict[str, torch.Tensor]:
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
                new_params[name].append(parameters[client][name].data)
            except:
                new_params[name] = []
                new_params[name].append(parameters[client][name].data)

    for name in new_params:
        # @TODO: Is .long() really required?
        new_params[name] = torch.median(torch.stack(new_params[name]), dim=0)[0]

    return new_params


if __name__ == "__main__":
    # test the median function
    params = {'1': {'1': torch.Tensor([[1, 2, 3], [4, 5, 6]]), '2': torch.Tensor([[7, 8, 9], [10, 11, 12]])},
              '2': {'1': torch.Tensor([[13, 14, 15], [16, 17, 18]]), '2': torch.Tensor([[19, 20, 21], [22, 23, 24]])},
              '3': {'1': torch.Tensor([[25, 26, 27], [28, 29, 30]]), '2': torch.Tensor([[31, 32, 33], [34, 35, 36]])}}
    sizes = {'1': 2, '2': 2, '3': 3}
    print(median(params, sizes))
    # test torch stack

    mal_list = np.random.choice([1,2,3], 0, replace=False)
    print(mal_list)

