import numpy as np


def get_mal_dataset(dataset, num_mal, num_classes):
    """
    Resturns a list of malicious datasets
    params: dataset: the original dataset
    params: num_mal: the number of malicious samples
    params: num_classes: the number of classes in the dataset
    """
    X_list = np.random.choice(len(dataset), num_mal)  # randomly select num_mal samples
    print("Get malicious images ID from dataset: ", X_list)
    Y_true = []
    for i in X_list:
        _, Y = dataset[i]
        Y_true.append(Y)
    Y_mal = []
    # randomly assign the malicious samples to different classes
    for i in range(num_mal):
        allowed_targets = list(range(num_classes))
        allowed_targets.remove(Y_true[i])
        # keep selected targeted same
        np.random.seed(i+100)
        Y_mal.append(np.random.choice(allowed_targets))
    return X_list, Y_mal, Y_true
