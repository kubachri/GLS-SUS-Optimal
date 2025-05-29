# src/model/builder.py

from pyomo.environ import ConcreteModel
from src.config import ModelConfig
from src.data.loader import load_data, load_techdata
from src.data.preprocess import scale_tech_parameters, slice_time_series
from src.model.sets import define_sets
from src.model.parameters import define_params
from src.model.variables import define_variables
from src.model.constraints import add_constraints
from src.model.objective import define_objective


def build_model(cfg: ModelConfig) -> ConcreteModel:
    """
    Build and return the Pyomo model based on the provided configuration.

    Parameters
    ----------
    cfg : ModelConfig
        Configuration object controlling test mode, horizon length, and penalty.

    Returns
    -------
    ConcreteModel
        The constructed and fully specified Pyomo optimization model.
    """
    # 1) Load raw data
    data    = load_data()
    tech_df = load_techdata()

    # 2) Preprocess technology parameters (scaling capacity, minima, ramp rates)
    data, tech_df = scale_tech_parameters(data, tech_df)

    # 3) Optionally slice time-series for test-mode runs
    if cfg.test_mode:
        data = slice_time_series(data, cfg.n_test)

    # 4) Assemble the model
    model = ConcreteModel()
    define_sets(model, data, tech_df)
    define_params(model, data, tech_df)
    define_variables(model)
    add_constraints(model)
    define_objective(model, penalty=cfg.penalty)

    return model
