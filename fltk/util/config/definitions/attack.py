from enum import unique, Enum


@unique
class Attack(Enum):
    fang = 'fang'
    lie = 'lie'
    minMax = 'min-max'
