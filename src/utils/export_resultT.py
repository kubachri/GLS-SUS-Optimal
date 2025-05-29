# src/utils/export_results.py

import pandas as pd
from pyomo.environ import value
from pathlib import Path

def export_results(model, path: str = None):
    """
    Export GAMS‐style ResultT, ResultF and ResultA tables to Excel.

    Sheets produced:
      - ResultT_all   (Operation, Volume, Costs_EUR, Startcost_EUR, Variable_OM_cost_EUR)
      - Flows         (areaFrom, areaTo, energy)
      - ResultA_all   (Buy, Sale, Demand, Import_price_EUR, Export_price_EUR, Buy_EUR, Sale_EUR)
    """
    # 1) Determine output path
    if path is None:
        project_root = Path(__file__).parents[2]
        output = project_root / "results" / "Results.xlsx"
    else:
        output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 2) Common time index
    times = list(model.T)
    time_cols = [str(t) for t in times]

    # --- build ResultT blocks ---
    pairs = set(model.f_in) | set(model.f_out)

    # 3a) Operation
    op = []
    for g, e in pairs:
        row = {'Result': 'Operation', 'tech': g, 'energy': e}
        for t in times:
            gen = value(model.Generation[g, e, t]) if (g, e) in model.f_out else 0
            use = value(model.Fueluse[g, e, t])      if (g, e) in model.f_in  else 0
            row[str(t)] = gen - use
        op.append(row)
    df_op = pd.DataFrame(op)

    # 3b) Volume
    vol = []
    for g in model.G_s:
        for e in (f for (gg, f) in model.f_out if gg == g):
            row = {'Result': 'Volume', 'tech': g, 'energy': e}
            for t in times:
                row[str(t)] = value(model.Volume[g, t])
            vol.append(row)
    df_vol = pd.DataFrame(vol)

    # 3c) Costs_EUR
    cost = []
    for g, e in pairs:
        row = {'Result': 'Costs_EUR', 'tech': g, 'energy': e}
        for t in times:
            imp_qty  = value(model.Fueluse[g, e, t])    if (g, e) in model.f_in  else 0
            sale_qty = value(model.Generation[g, e, t]) if (g, e) in model.f_out else 0
            imp_price  = sum(model.price_buy[a, e, t]  for a in model.A if (a, e) in model.buyE)
            sale_price = sum(model.price_sale[a, e, t] for a in model.A if (a, e) in model.saleE)
            row[str(t)] = imp_qty * imp_price - sale_qty * sale_price
        cost.append(row)
    df_cost = pd.DataFrame(cost)

    # 3d) Startcost_EUR
    start = []
    for g in model.G:
        row = {'Result': 'Startcost_EUR', 'tech': g, 'energy': 'system_cost'}
        for t in times:
            row[str(t)] = value(model.Startcost[g, t])
        start.append(row)
    df_start = pd.DataFrame(start)

    # 3e) Variable_OM_cost_EUR
    varom = []
    for g in model.G:
        row = {'Result': 'Variable_OM_cost_EUR', 'tech': g, 'energy': 'system_cost'}
        for t in times:
            row[str(t)] = value(model.Fuelusetotal[g, t]) * model.cvar[g]
        varom.append(row)
    df_varom = pd.DataFrame(varom)

    # sort each block by tech→energy
    for df in (df_op, df_vol, df_cost, df_start, df_varom):
        df.sort_values(['tech','energy'], inplace=True)

    # concatenate all ResultT
    df_T = pd.concat([df_op, df_vol, df_cost, df_start, df_varom], ignore_index=True)
    df_T = df_T[['Result','tech','energy'] + time_cols]

    # --- build Flows sheet ---
    flows = []
    for ao, ai, f in model.flowset:
        row = {'areaFrom': ao, 'areaTo': ai, 'energy': f}
        for t in times:
            row[str(t)] = value(model.Flow[ao, ai, f, t])
        flows.append(row)
    df_F = pd.DataFrame(flows)
    df_F.sort_values(['areaFrom','areaTo','energy'], inplace=True)
    df_F = df_F[['areaFrom','areaTo','energy'] + time_cols]

    # --- build ResultA sheet ---
    A_rows = []

    # 1) Buy & 2) Sale quantities
    for res, varset in (('Buy',  model.buyE), ('Sale', model.saleE)):
        for a, e in varset:
            row = {'Result': res, 'area': a, 'energy': e}
            for t in times:
                row[str(t)] = value(model.Buy[a, e, t]) if res == 'Buy' else value(model.Sale[a, e, t])
            A_rows.append(row)

    # 3) Demand – only truly initialized & non‐zero
    raw_demand = dict(model.demand.items())  # only contains explicitly set entries
    # pick out (area,energy) combos that have at least one non‐zero hour
    dem_pairs = sorted({
        (a, e)
        for (a, e, t), val in raw_demand.items()
        if val != 0
    })
    for a, e in dem_pairs:
        row = {'Result': 'Demand', 'area': a, 'energy': e}
        for t in times:
            row[str(t)] = raw_demand.get((a, e, t), 0)
        A_rows.append(row)

    # 4) Import_price_EUR & 5) Export_price_EUR
    for res, price_param, sel in (
        ('Import_price_EUR',  model.price_buy,  model.buyE),
        ('Export_price_EUR',  model.price_sale, model.saleE)
    ):
        for a, e in sel:
            row = {'Result': res, 'area': a, 'energy': e}
            for t in times:
                row[str(t)] = price_param[a, e, t]
            A_rows.append(row)

    # 6) Buy_EUR & 7) Sale_EUR
    for res, varset, price_param in (
        ('Buy_EUR',  model.buyE,  model.price_buy),
        ('Sale_EUR', model.saleE, model.price_sale)
    ):
        for a, e in varset:
            row = {'Result': res, 'area': a, 'energy': e}
            for t in times:
                qty   = value(model.Buy[a, e, t])  if res == 'Buy_EUR'  else value(model.Sale[a, e, t])
                price = price_param[a, e, t]
                row[str(t)] = qty * price
            A_rows.append(row)

    df_A = pd.DataFrame(A_rows)

    # enforce your exact Result‐block order
    block_order = [
        'Buy', 'Sale', 'Demand',
        'Import_price_EUR', 'Export_price_EUR',
        'Buy_EUR', 'Sale_EUR'
    ]
    df_A['Result'] = pd.Categorical(df_A['Result'], categories=block_order, ordered=True)
    df_A.sort_values(['Result','area','energy'], inplace=True)
    df_A = df_A[['Result','area','energy'] + time_cols]

    # --- write everything in one workbook ---
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_T.to_excel(writer, sheet_name='ResultT_all', index=False)
        df_F.to_excel(writer, sheet_name='Flows',      index=False)
        df_A.to_excel(writer, sheet_name='ResultA_all',index=False)

    print(f"Wrote ResultT, Flows and ResultA to {output.resolve()}")
