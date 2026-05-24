# Decomp: three-branch additively decomposable composed retrieval
# Sibling package to `sira/`. Operates on the cache/cirr_cls/ CLS-level cache.

from .model import DecompModel
from .losses import SymInfoNCE
from .dataset import CIRRClsTriplet, CIRRClsQuery, load_gallery

__all__ = [
    "DecompModel",
    "SymInfoNCE",
    "CIRRClsTriplet",
    "CIRRClsQuery",
    "load_gallery",
]
