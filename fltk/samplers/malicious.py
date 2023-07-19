import numpy as np

from fltk.samplers import DistributedSamplerWrapper


class MaliciousSampler(DistributedSamplerWrapper):
    """
    Distributed Sampler implementation that samples uniformly from the available datapoints, assuming all clients
    have an equal distribution over the data (following the original random seed).
    """
    def __init__(self, dataset, mal_samples, num_replicas=None, rank=None, seed=0):
        super().__init__(dataset, num_replicas=2, rank=0, seed=seed)
        indices = list(range(len(self.dataset)))
        # randomly choose mal_samples indices
        self.indices = np.random.choice(indices, size=mal_samples, replace=False)

# debugging
