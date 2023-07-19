from fltk.util.config.definitions.aggregate import Aggregations
from . import bulyan
from .fed_avg import fed_avg
from  .median import median
from .trmean import trimmed_mean
from .bulyan import bulyan
from .krum import krum
from .multiKrum import multiKrum
def get_aggregation(name: Aggregations):
    """
    Helper function to get specific Aggregation class references.
    @param name: Aggregation class reference.
    @type name: Aggregations
    @return: Class reference corresponding to the requested Aggregation definition.
    @rtype: Type[torch.optim.Optimizer]
    """
    enum_type = Aggregations(name.value)
    aggregations_dict = {
            Aggregations.fedavg: fed_avg,
            Aggregations.median: median,
            Aggregations.trmean: trimmed_mean,
            Aggregations.sum: lambda x: x,
            Aggregations.avg: lambda x: x*2,
            Aggregations.bulyan: bulyan,
            Aggregations.krum: krum,
            Aggregations.multiKrum: multiKrum
        }
    return aggregations_dict[enum_type]
