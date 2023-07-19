import torch
def lie_nn_parameters(dict_parameters, args = None):
    """
    generate lie parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """
    z_value = 1.3
    mean_params = {}
    std_params = {}
    for name in dict_parameters[list(dict_parameters.keys())[0]].keys():

        mean_params[name] = sum([param[name].data for param in dict_parameters.values()]).data/len(dict_parameters.values())

        for key in dict_parameters.keys():
            mean_params[name] = mean_params[name].data + dict_parameters[key][name].data
        mean_params[name] = mean_params[name].data / len(dict_parameters.keys())

        _std_params = []
        for param in dict_parameters.values():
            _std_params.append(param[name].data)
        val = torch.stack(_std_params).data
        std_params[name] = torch.std(val.float(), 0).data

    # mean_dis = model_distance(mean_params, dict_parameters[list(dict_parameters.keys())[0]])
    # print("lie mean dis:", mean_dis)
    mal_param = {}
    for name in mean_params.keys():
        mal_param[name] = mean_params[name].data + z_value * std_params[name].data
    return mal_param

if __name__ == '__main__':
    # test lie_attack
    benign_client_gradients = {'client_1': {'layer_1': torch.tensor([[1., 2., 3.], [4., 5., 6.]]), 'layer_2': torch.tensor([[7., 8., 9.], [10., 11., 12.]])},
                                'client_2': {'layer_1': torch.tensor([[13., 14., 15.], [16., 17., 18.]]), 'layer_2': torch.tensor([[19., 20., 21.], [22., 23., 24.]])}}
    last_model = {'layer_1': torch.tensor([[1., 2., 3.], [4., 5., 6.]]), 'layer_2': torch.tensor([[7., 8., 9.], [10., 11., 12.]])}
    mal_weights = lie_nn_parameters(benign_client_gradients)
    print(mal_weights)