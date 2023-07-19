import torch

def model_distance(m1_params, m2_params):
    """
    for ndss_nn_parameters
    """
    distance = 0
    for key in m1_params.keys():
        # print(m1_params[key].type())
        distance = distance + (torch.norm(m1_params[key].float()-m2_params[key].float())**2)
    return distance


def get_deviation_and_model_avg(dict_parameters, normal_idx_list, deviation_type):
    """
    for ndss_nn_parameters
    """
    model_avg = {}
    for normal_idx in normal_idx_list:
        client_param = dict_parameters[normal_idx]  # normal client
        for key in client_param.keys():
            if key in model_avg:
                model_avg[key] = model_avg[key].data + client_param[key].data
            else:
                model_avg[key] = client_param[key]
    deviation = {}

    print("deviation_type!!!:", deviation_type)

    for key in model_avg.keys():
        model_avg[key] = model_avg[key].data / len(normal_idx_list)
        if deviation_type == "sign":
            deviation[key] = torch.sign(model_avg[key])

    if deviation_type == "std":
        for key in model_avg.keys():
            _std_params = []
            for param in dict_parameters.values():
                _std_params.append(param[key].data)
            val = torch.stack(_std_params).data
            deviation[key] = torch.std(val.float(), 0).data

    return model_avg, deviation


def get_malicious_model(model_avg, lamda, deviation):
    """
    for ndss_nn_parameters
    """
    mal_model = {}
    for key in model_avg.keys():
        mal_model[key] = model_avg[key].data - (lamda * deviation[key].data)
    return mal_model


def oracle_check(mal_model, dict_parameters, normal_idx_list, upper_bound):
    """
    for ndss_nn_parameters
    """
    for c in normal_idx_list:
        mal_dis = model_distance(mal_model, dict_parameters[c])
        if mal_dis > upper_bound:
            return False
        # else:
        #     print("mal_dis", mal_dis, "upper_bound", upper_bound)
    return True

def ndss_nn_parameters(dict_parameters, args='std'):
    """
    The implementation of "Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses
    for Federated Learning", (AGR-agnostic attacks, Min-Max), according the authors' open-sourced code:
    https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning/blob/main/cifar10/release-fedsgd-alexnet-mkrum-unknown-benign-gradients.ipynb

    :param parameters: nn model named parameters
    :type parameters: list
    """

    normal_idx_list = list(dict_parameters.keys())
    # calculate the upper bound of gradient distance
    upper_bound = 0
    for i in range(len(normal_idx_list)-1):
        for j in range(i+1, len(normal_idx_list)):
            dis = model_distance(dict_parameters[normal_idx_list[i]], dict_parameters[normal_idx_list[j]])
            if dis > upper_bound:
                upper_bound = dis
    print("upper bound is:", upper_bound)

    model_avg, deviation = get_deviation_and_model_avg(dict_parameters, normal_idx_list, args)


    # building malicious model parameters for the attacker
    lamda = torch.Tensor([10.0]).float()
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_model = get_malicious_model(model_avg, lamda, deviation)
        # print('lamda is ', lamda)
        if oracle_check(mal_model, dict_parameters, normal_idx_list, upper_bound):
            # print('successful lamda is ', lamda)

            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_model = get_malicious_model(model_avg, lamda_succ, deviation)
    return mal_model

if __name__ == '__main__':
    # test lie_attack
    benign_client_gradients = {'client_2': {'layer_1': torch.tensor([[100., 2., 3.], [4., 5., 6.]]), 'layer_2': torch.tensor([[7., 8., 9.], [10., 11., 12.]])},
                                'client_3': {'layer_1': torch.tensor([[13., 14., 15.], [16., 17., 18.]]), 'layer_2': torch.tensor([[19., 20., 21.], [22., 23., 24.]])},
                                'client_8': {'layer_1': torch.tensor([[25., 26., 27.], [28., 29., 30.]]), 'layer_2': torch.tensor([[31., 32., 33.], [34., 35., 36.]])},
                                'client_4': {'layer_1': torch.tensor([[37., 38., 39.], [40., 41., 42.]]), 'layer_2': torch.tensor([[43., 44., 45.], [46., 47., 48.]])},
                                'client_5': {'layer_1': torch.tensor([[49., 50., 51.], [52., 53., 54.]]), 'layer_2': torch.tensor([[55., 56., 57.], [58., 59., 60.]])}}

    mal_weights = ndss_nn_parameters(benign_client_gradients)
    print(mal_weights)