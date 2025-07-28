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
    if path is None:
        project_root = Path(__file__).parents[2]
        filename = "Inputs.xlsx"
        if cfg.test_mode:
            filename = "test_" + filename
        out = project_root / "results" / filename
    else:
        out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = pd.ExcelWriter(out, engine='xlsxwriter')

    # Export Sets
    for s in model.component_objects(Set, descend_into=True):
        rows = [(member if isinstance(member, tuple) else (member,)) for member in s]
        if not rows:
            continue
        dim = len(rows[0])
        col_names = [f"{s.name}_idx{i+1}" for i in range(dim)]
        df = pd.DataFrame(rows, columns=col_names)
        df.to_excel(writer, sheet_name=f"Set__{s.name}", index=False)

    # Export Params
    for p in model.component_objects(Param, descend_into=True):
        pname = p.name
        rows = []

        # ✅ Only iterate over defined keys
        for idx in p:
            idx_tup = idx if isinstance(idx, tuple) else (idx,)
            try:
                v = value(p[idx_tup])
            except:
                v = np.nan
            rows.append(idx_tup + (v,))

        if not rows:
            continue

        # Special reshape for price_buy / price_sale
        if pname in {"price_buy", "price_sale", "InterconnectorCapacity", "demand"}:
            df = pd.DataFrame(rows, columns=["Area", "Fuel", "Time", pname])
            df = df[df[pname].notna() & (df[pname] != 0)]  # ✅ filter only defined nonzero
            df["Area.Fuel"] = df["Area"] + "." + df["Fuel"]
            df_pivot = df.pivot(index="Time", columns="Area.Fuel", values=pname)
            df_pivot.reset_index(inplace=True)
            df_pivot.rename(columns={"Time": "Hour"}, inplace=True)
            short_name = pname if len(pname) <= 25 else pname[:25]
            df_pivot.to_excel(writer, sheet_name=f"{short_name}_pivoted", index=False)
            continue

        # Generic param export
        num_idx = len(rows[0]) - 1
        col_names = [f"{pname}_idx{i+1}" for i in range(num_idx)] + [pname]
        df = pd.DataFrame(rows, columns=col_names)
        df.to_excel(writer, sheet_name=f"Param__{pname}", index=False)

    writer.close()
    print(f"✔ All inputs exported to {out}")
