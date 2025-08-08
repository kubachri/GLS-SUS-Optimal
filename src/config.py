# src/config.py
from dataclasses import dataclass
import os

@dataclass
class ModelConfig:
    test_mode:      bool    = False   # run a short “N_test” horizon?
    n_test:         int     = 168     # number of hours in test mode
    penalty:        float   = 100000     # slack‐penalty in objective
    data:           str     = 'inc_data'  # name of folder under project root
    demand_target:  bool    = False
    carbon_tax:     int     = 150
    sensitivity:    bool    = False

    @property
    def data_dir(self) -> str:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        return os.path.join(root, self.data)