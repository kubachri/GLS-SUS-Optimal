import pandas as pd
from pyomo.core.base.param import Param
from pyomo.core.base.set import Set
from pyomo.environ import value
from pathlib import Path
import numpy as np


def export_inputs(model, cfg, path: str = None):
    """
    Export all Sets and Params of `model` into an Excel workbook.
    """
    # 1) Determine output path
    if path is None:
        project_root = Path(__file__).parents[2]
        out = project_root / "results" / "Inputs.xlsx"
    else:
        out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = pd.ExcelWriter(out, engine='xlsxwriter')

    # 2) Export every Set
    for s in model.component_objects(Set, descend_into=True):
        # 1) build a list of rows, each row is a tuple of indices
        rows = []
        for member in s:
            # make everything a tuple, even if singleton
            tup = member if isinstance(member, tuple) else (member,)
            rows.append(tup)

        if not rows:
            # empty set? skip it (or create an empty DataFrame if you prefer)
            continue

        # 2) figure out how many dims this Set has
        dim = len(rows[0])

        # 3) make column names like MySet_idx1, MySet_idx2, …
        col_names = [f"{s.name}_idx{i+1}" for i in range(dim)]

        # 4) create the DataFrame with the right shape
        df = pd.DataFrame(rows, columns=col_names)

        # 5) write to Excel (one sheet per Set)
        sheet = f"Set__{s.name}"
        df.to_excel(writer, sheet_name=sheet, index=False)

    # 3) Export every Param
    for p in model.component_objects(Param, descend_into=True):
        rows = []
        for idx in p.index_set():
            # normalize idx to a tuple
            idx_tup = idx if isinstance(idx, tuple) else (idx,)
            try:
                v = value(p[idx_tup])
            except ValueError:
                # undefined param entry → mark as NaN
                v = np.nan
            rows.append(idx_tup + (v,))

        if not rows:
            # nothing to write for this Param
            continue

        # infer how many idx dimensions we have:
        num_idx = len(rows[0]) - 1

        # build exactly num_idx names plus the value column
        col_names = [f"{p.name}_idx{i+1}" for i in range(num_idx)] + [p.name]

        df = pd.DataFrame(rows, columns=col_names)
        sheet = f"Param__{p.name}"
        df.to_excel(writer, sheet_name=sheet, index=False)

    writer.close()
    print(f"✔ All inputs exported to {out}")
