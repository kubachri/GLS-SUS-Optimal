# src/utils/export_resultT.py

import pandas as pd
from pyomo.environ import value
from pathlib import Path

def export_results(model, path: str = None):
    """
    Export the five GAMS‐style ResultT tables to Excel, one row per (Result,tech,energy).

    Sheets produced:
      - Operation
      - Volume
      - Costs_EUR
      - Startcost_EUR
      - Variable_OM_cost_EUR

    Parameters
    ----------
    model : ConcreteModel
        A solved Pyomo model with sets T, f_in, f_out, G_s, A, buyE, saleE and
        vars Generation, Fueluse, Volume, Startcost, Fuelusetotal, and params price_buy, price_sale, cvar.
    path : str, optional
        Path to write the .xlsx; defaults to <project_root>/results/ResultT.xlsx.
    """
    # 1) Determine output path
    if path is None:
        project_root = Path(__file__).parents[2]
        output = project_root / "results" / "ResultT.xlsx"
    else:
        output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 2) Common time index
    times = list(model.T)
    time_cols = [str(t) for t in times]

    # 3) Build each table

    # --- 3a) Operation = Generation - Fueluse ---
    op_rows = []
    pairs = set(model.f_in) | set(model.f_out)
    for g, e in pairs:
        # Only export if imported or exported at all
        row = {'Result': 'Operation', 'tech': g, 'energy': e}
        for t in times:
            gen = value(model.Generation[g, e, t]) if (g, e) in model.f_out else 0
            use = value(model.Fueluse[g, e, t])      if (g, e) in model.f_in  else 0
            row[str(t)] = gen - use
        op_rows.append(row)
    df_op = pd.DataFrame(op_rows, columns=['Result','tech','energy'] + time_cols)

    # --- 3b) Volume = storage Volume for storage techs only ---
    vol_rows = []
    for g in model.G_s:
        # find its export‐fuel(s)
        fuels = [f for (gg, f) in model.f_out if gg == g]
        for f in fuels:
            row = {'Result': 'Volume', 'tech': g, 'energy': f}
            for t in times:
                row[str(t)] = value(model.Volume[g, t])
            vol_rows.append(row)
    df_vol = pd.DataFrame(vol_rows, columns=['Result','tech','energy'] + time_cols)

    # --- 3c) Costs_EUR = import_cost*Fueluse - export_rev*Generation ---
    cost_rows = []
    for g, e in pairs:
        row = {'Result': 'Costs_EUR', 'tech': g, 'energy': e}
        for t in times:
            imp_qty  = value(model.Fueluse[g, e, t])    if (g, e) in model.f_in  else 0
            sale_qty = value(model.Generation[g, e, t]) if (g, e) in model.f_out else 0
            # sum prices across areas
            imp_price  = sum(model.price_buy[a, e, t]  for a in model.A if (a,e) in model.buyE)
            sale_price = sum(model.price_sale[a, e, t] for a in model.A if (a,e) in model.saleE)
            row[str(t)] = imp_qty * imp_price - sale_qty * sale_price
        cost_rows.append(row)
    df_cost = pd.DataFrame(cost_rows, columns=['Result','tech','energy'] + time_cols)

    # --- 3d) Startcost_EUR = Startcost var, tagged as system_cost ---
    start_rows = []
    for g in model.G:
        row = {'Result': 'Startcost_EUR', 'tech': g, 'energy': 'system_cost'}
        for t in times:
            row[str(t)] = value(model.Startcost[g, t])
        start_rows.append(row)
    df_start = pd.DataFrame(start_rows, columns=['Result','tech','energy'] + time_cols)

    # --- 3e) Variable_OM_cost_EUR = Fuelusetotal * cvar, system_cost ---
    varom_rows = []
    for g in model.G:
        row = {'Result': 'Variable_OM_cost_EUR', 'tech': g, 'energy': 'system_cost'}
        for t in times:
            row[str(t)] = value(model.Fuelusetotal[g, t]) * model.cvar[g]
        varom_rows.append(row)
    df_varom = pd.DataFrame(varom_rows, columns=['Result','tech','energy'] + time_cols)

    # 4) Sort each block by tech→energy
    for df in (df_op, df_vol, df_cost, df_start, df_varom):
        df.sort_values(['tech','energy'], inplace=True)

    # 5) Concatenate in the exact order you want:
    all_df = pd.concat(
        [df_op, df_vol, df_cost, df_start, df_varom],
        ignore_index=True
    )

    # 6) Write a single sheet
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        all_df.to_excel(writer, sheet_name='ResultT_all', index=False)

    print(f"Wrote all ResultT tables to one sheet in {output.resolve()}")

    # --- Build the Flows sheet (ResultF) ---
    flow_rows = []
    for area_out, area_in, energy in model.flowset:
        row = {
            'areaFrom': area_out,
            'areaTo':   area_in,
            'energy':   energy,
        }
        for t in times:
            row[str(t)] = value(model.Flow[area_out, area_in, energy, t])
        flow_rows.append(row)

    df_flow = pd.DataFrame(
        flow_rows,
        columns=['areaFrom','areaTo','energy'] + time_cols
    )
    # sort by from→to→energy
    df_flow.sort_values(['areaFrom','areaTo','energy'], inplace=True)

    # --- Write both sheets in one workbook ---
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # your combined ResultT table
        all_df.to_excel(writer, sheet_name='ResultT_all', index=False)
        # the new Flows table
        df_flow.to_excel(writer, sheet_name='Flows',      index=False)

    print(f"Wrote both ResultT_all and Flows sheets to {output.resolve()}")
