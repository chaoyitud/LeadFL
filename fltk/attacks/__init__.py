from fltk.util.config.definitions.attack import Attack
from .lie_attack import lie_nn_parameters
from .fang_attack import fang_nn_parameters
from .ndss_attack import ndss_nn_parameters
def get_attack(name: Attack):
    """
    Helper function to get specific Aggregation class references.
    @param name: Aggregation class reference.
    @type name: Aggregations
    @return: Class reference corresponding to the requested Aggregation definition.
    @rtype: Type[torch.optim.Optimizer]
    """
    enum_type = Attack(name.value)
    attacks_dict = {
        Attack.fang: fang_nn_parameters,
        Attack.lie: lie_nn_parameters,
        Attack.minMax: ndss_nn_parameters,
        }
    return attacks_dict[enum_type]
