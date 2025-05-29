# src/config.py
from dataclasses import dataclass

@dataclass
class ModelConfig:
    test_mode: bool   = False   # run a short “N_test” horizon?
    n_test:    int    = 168     # number of hours in test mode
    penalty:   float  = 1e6     # slack‐penalty in objective
