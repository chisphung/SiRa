# CIRR (Composed Image Retrieval on Real-life images) integration for SIRA
# Provides dataset loaders, precomputation, and evaluation utilities.

from .cirr_dataset import (
    CIRRCLSDataset,
    CIRRCLSTestDataset,
    PrecomputedCIRRCLSDataset,
)

__all__ = [
    "CIRRCLSDataset",
    "CIRRCLSTestDataset",
    "PrecomputedCIRRCLSDataset",
]
