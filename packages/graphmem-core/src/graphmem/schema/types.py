from enum import Enum


class Layer(str, Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


class EdgeType(str, Enum):
    DERIVED_FROM = "DERIVED_FROM"
    MENTIONS = "MENTIONS"
    RELATES_TO = "RELATES_TO"
    DEPENDS_ON = "DEPENDS_ON"
    CONTRADICTS = "CONTRADICTS"
    SUPERSEDES = "SUPERSEDES"
    PART_OF = "PART_OF"
