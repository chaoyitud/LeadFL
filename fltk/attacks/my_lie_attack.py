import torch
def lie_attack(benign_client_gradients: dict, last_model:dict, z:float = 1.3) -> dict:
    """
    Perform the lie attacks.
    :param benign_client_gradients:
    :return: mal_gradients:
    """
    benign_client_gradients_stack = {}
    for client_name in benign_client_gradients:
        for layer_name in benign_client_gradients[client_name]:
            benign_client_gradients_stack[layer_name] = [] if layer_name not in benign_client_gradients_stack else benign_client_gradients_stack[layer_name]
            benign_client_gradients_stack[layer_name].append(benign_client_gradients[client_name][layer_name])
    for layer_name in benign_client_gradients_stack:
        benign_client_gradients_stack[layer_name] = torch.stack(benign_client_gradients_stack[layer_name])
    benign_client_gradients_mean = {}
    benign_client_gradients_std = {}
    for layer_name in benign_client_gradients_stack:
        # if type is long, change to float
        if benign_client_gradients_stack[layer_name].dtype == torch.long:
            benign_client_gradients_stack[layer_name] = benign_client_gradients_stack[layer_name].float()
        benign_client_gradients_mean[layer_name] = benign_client_gradients_stack[layer_name].mean(dim=0)
        benign_client_gradients_std[layer_name] = benign_client_gradients_stack[layer_name].std(dim=0)
    mal_gradients = {}
    mal_weights = {}
    for layer_name in benign_client_gradients_mean:
        mal_gradients[layer_name] = z * benign_client_gradients_std[layer_name] + benign_client_gradients_mean[layer_name]
        mal_weights[layer_name] = mal_gradients[layer_name] + last_model[layer_name]
    return mal_gradients, mal_weights

if __name__ == '__main__':
    # test lie_attack
    benign_client_gradients = {'client_1': {'layer_1': torch.tensor([[1., 2., 3.], [4., 5., 6.]]), 'layer_2': torch.tensor([[7., 8., 9.], [10., 11., 12.]])},
                                'client_2': {'layer_1': torch.tensor([[13., 14., 15.], [16., 17., 18.]]), 'layer_2': torch.tensor([[19., 20., 21.], [22., 23., 24.]])}}
    last_model = {'layer_1': torch.tensor([[1., 2., 3.], [4., 5., 6.]]), 'layer_2': torch.tensor([[7., 8., 9.], [10., 11., 12.]])}
    mal_gradients, mal_weights = lie_attack(benign_client_gradients, last_model)
    print(mal_gradients)