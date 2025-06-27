# src/utils/export_results.py

import pandas as pd
from pyomo.environ import value
from pathlib import Path
from src.config import ModelConfig

def export_results(model, cfg: ModelConfig, path: str = None):
    """
    Export GAMS‐style ResultT, ResultF and ResultA tables to Excel,
    plus “ResultTsum”, “ResultFsum” and “ResultAsum” pivoted summaries.

    Sheets produced:
      - ResultT_all    (hourly: Operation, Volume, Costs_EUR, Startcost_EUR, Variable_OM_cost_EUR)
      - ResultTsum     (pivot: one row per (Result, tech), columns = each energy, values = sum over hours;
                        “Result” in the same block‐order as the hourly sheet)
      - Flows          (hourly: areaFrom, areaTo, energy)
      - ResultFsum     (pivot: one row per (areaFrom, areaTo), columns = each energy, values = sum over hours)
      - ResultA_all    (hourly: Buy, Sale, Demand, Import_price_EUR, Export_price_EUR, Buy_EUR, Sale_EUR)
      - ResultAsum     (pivot: one row per (Result, area), columns = each energy,
                        values = sum over hours except for price rows:
                        – for energy=="Electricity", take the average over all hours
                        – for all other energies, take the first‐hour price)
      - ResultC        (hourly capacity factors + summary below)
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
    ntimes = len(times)

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

    # sort each block by tech → energy
    for df in (df_op, df_vol, df_cost, df_start, df_varom):
        df.sort_values(['tech','energy'], inplace=True)

    # concatenate all ResultT (hourly)
    df_T = pd.concat([df_op, df_vol, df_cost, df_start, df_varom], ignore_index=True)
    df_T = df_T[['Result','tech','energy'] + time_cols]

    # --- build Flows sheet (hourly) ---
    flows = []
    for ao, ai, f in model.flowset:
        row = {'areaFrom': ao, 'areaTo': ai, 'energy': f}
        for t in times:
            row[str(t)] = value(model.Flow[ao, ai, f, t])
        flows.append(row)
    df_F = pd.DataFrame(flows)
    df_F.sort_values(['areaFrom','areaTo','energy'], inplace=True)
    df_F = df_F[['areaFrom','areaTo','energy'] + time_cols]

    # ------------------------------------------------
    # --- RESULTTsum: pivot AND enforce block order ---
    # ------------------------------------------------
    df_Tsum = (
        df_T
          .set_index(['Result','tech','energy'])[time_cols]
          .sum(axis=1)             # sum each (Result,tech,energy) over all t
          .unstack(fill_value=0)   # pivot so “energy” becomes columns
          .reset_index()           # bring “Result” & “tech” back as columns
    )

    # Enforce the exact block‐order for “Result” in df_Tsum
    block_order_T = [
        'Operation',
        'Volume',
        'Costs_EUR',
        'Startcost_EUR',
        'Variable_OM_cost_EUR'
    ]
    df_Tsum['Result'] = pd.Categorical(
        df_Tsum['Result'],
        categories=block_order_T,
        ordered=True
    )
    df_Tsum.sort_values(['Result','tech'], inplace=True)


    # --- build ResultA sheet (hourly) ---
    A_rows = []

    # 1) Buy & 2) Sale quantities
    for res, varset in (('Buy',  model.buyE), ('Sale', model.saleE)):
        for a, e in varset:
            row = {'Result': res, 'area': a, 'energy': e}
            for t in times:
                row[str(t)] = (
                    value(model.Buy[a, e, t]) if res == 'Buy'
                    else value(model.Sale[a, e, t])
                )
            A_rows.append(row)

    # 3) Demand – only truly initialized & non‐zero
    raw_demand = dict(model.demand.items())
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
                qty   = (
                    value(model.Buy[a, e, t])
                    if res == 'Buy_EUR'
                    else value(model.Sale[a, e, t])
                )
                price = price_param[a, e, t]
                row[str(t)] = qty * price
            A_rows.append(row)

    df_A = pd.DataFrame(A_rows)

    # enforce the exact block‐order for df_A
    block_order_A = [
        'Buy',
        'Sale',
        'Demand',
        'Import_price_EUR',
        'Export_price_EUR',
        'Buy_EUR',
        'Sale_EUR'
    ]
    df_A['Result'] = pd.Categorical(
        df_A['Result'],
        categories=block_order_A,
        ordered=True
    )
    df_A.sort_values(['Result','area','energy'], inplace=True)
    df_A = df_A[['Result','area','energy'] + time_cols]


    # ------------------------------------------------------------
    # --- RESULTAsum: pivot but adjust “Electricity” price rows ---
    # ------------------------------------------------------------
    price_mask = df_A['Result'].isin(['Import_price_EUR','Export_price_EUR'])

    # (a) Build a small DataFrame that holds exactly one price‐value per
    #     (Result, area, energy):
    price_rows = df_A[price_mask].copy()

    # Compute “PriceValue” as:
    #  - If energy == "Electricity", average over all hours
    #  - Else, take the first hour’s price (time_cols[0])
    price_rows['PriceValue'] = price_rows.apply(
        lambda row: row[time_cols].mean()
                    if row['energy'] == 'Electricity'
                    else row[time_cols[0]],
        axis=1
    )

    # Pivot this “PriceValue” table so that each “energy” becomes its own column:
    df_price = (
        price_rows
          .set_index(['Result','area','energy'])['PriceValue']
          .unstack(level='energy', fill_value=0)
          .reset_index()
    )
    # Now df_price has columns:
    #    [ 'Result', 'area', '<energy1>', '<energy2>', … ]
    # where for (Import_price_EUR, "DK1", "Electricity"), the cell is
    # the *average* of all hourly Import_price_EUR["DK1","Electricity",t].

    # (b) Build the non‐price sums exactly as before:
    df_nonprice = (
        df_A[~price_mask]
          .set_index(['Result','area','energy'])[time_cols]
          .sum(axis=1)             # sum each (Result,area,energy) over all t
          .unstack(fill_value=0)   # pivot so “energy” becomes columns
          .reset_index()           # bring “Result” & “area” back as columns
    )

    # (c) Combine them and re‐apply block order
    df_Asum = pd.concat([df_nonprice, df_price], ignore_index=True)
    df_Asum['Result'] = pd.Categorical(
        df_Asum['Result'],
        categories=block_order_A,
        ordered=True
    )
    df_Asum.sort_values(['Result','area'], inplace=True)

    if model.Demand_Target:
        # --- Duals sheet: 1) hourly CO2, 2) weekly methanol duals ---
        # 1) Hourly CO2 duals
        co2_rows = []
        for t in times:
            row = {'Time': str(t)}
            for area in ['Skive']:
                idx = (area, 'CO2', t)
                if idx in model.Balance.index_set():
                    dual_val = model.dual.get(model.Balance[idx], 0.0)
                else:
                    dual_val = 0.0
                row[f"CO2_{area}"] = dual_val
            co2_rows.append(row)
        df_co2 = pd.DataFrame(co2_rows)

        # 2) Weekly methanol‐target duals
        meth_rows = []
        for w in sorted(model.W):
            # constraint index is just w
            dual_val = model.dual.get(model.WeeklyMethanolTarget[w], 0.0)
            meth_rows.append({'Week': w, 'MethanolDual': dual_val})
        df_meth = pd.DataFrame(meth_rows)


    # ----------------------------------------------------------------
    # --- ResultC (capacity factors) ---------------------------------
    # ----------------------------------------------------------------
    C_rows = []
    for tech in model.G:
        cap = value(model.capacity[tech])
        row = {'Result': 'CapacityFactor', 'tech': tech}
        fuels = [e for (g,e) in model.f_out if g == tech]
        for t in times:
            gen_sum = sum(value(model.Generation[tech, e, t]) for e in fuels)
            row[str(t)] = gen_sum / cap if cap != 0 else 0
        C_rows.append(row)

    df_C_hourly = pd.DataFrame(C_rows, columns=['Result','tech'] + time_cols)

    summary_cf = []
    summary_flh = []
    for tech in model.G:
        cap = value(model.capacity[tech])
        fuels = [e for (g,e) in model.f_out if g == tech]
        total_gen = sum(
            value(model.Generation[tech, e, t])
            for e in fuels for t in times
        )
        avg_cf = total_gen / (cap * ntimes) if cap != 0 else 0
        summary_cf.append({'Result': 'CapacityFactor_Summary','tech': tech,'Average_CF': avg_cf})
        summary_flh.append({'Result': 'FullLoadHours','tech': tech,'FLH': avg_cf * ntimes})

    df_C_summary = pd.DataFrame(summary_cf + summary_flh, columns=['Result','tech','Average_CF','FLH'])

    # 9) Objective decomposition (total over all time‐steps, by element)
    decomp = []

    #  a) Fuel imports (“Buy_…”) are costs → negative contributions
    for (a, e) in model.buyE:
        tot = sum(
            value(model.price_buy[a, e, t] * model.Buy[a, e, t])
            for t in times
        )
        decomp.append({
            "Element": f"Buy_{e}",
            "Contribution": -tot
        })

    #  b) Fuel sales (“Sell_…”) are revenues → positive
    for (a, e) in model.saleE:
        tot = sum(
            value(model.price_sale[a, e, t] * model.Sale[a, e, t])
            for t in times
        )
        decomp.append({
            "Element": f"Sell_{e}",
            "Contribution": tot
        })

    #  c) Variable O&M on tech→energy
    tot_varom = sum(
        value(model.Generation[g, e, t] * model.cvar[g])
        for (g, e) in model.TechToEnergy
        for t in times
    )
    decomp.append({"Element": "Variable_OM", "Contribution": -tot_varom})

    #  d) Startup costs
    tot_start = sum(
        value(model.Startcost[g, t])
        for g in model.G
        for t in times
    )
    decomp.append({"Element": "Startup", "Contribution": -tot_start})

    # e) Slack penalties (skip any un‐initialized vars)
    tot_slack_imp = 0.0
    for (a, e, t) in model.DemandSet:
        var = model.SlackDemandImport[a, e, t]
        if var.value is not None:
            tot_slack_imp += value(var)

    tot_slack_exp = 0.0
    for (a, e, t) in model.DemandSet:
        var = model.SlackDemandExport[a, e, t]
        if var.value is not None:
            tot_slack_exp += value(var)

    penalty = cfg.penalty

    print('slackImport ', tot_slack_imp)
    print('slackExport ', tot_slack_exp)
    print('penalty ', penalty)

    decomp.append({
        "Element": "Slack",
        "Contribution": -penalty * (tot_slack_imp + tot_slack_exp)
    })

    df_decomp = pd.DataFrame(decomp)

    # ----------------------------------------------------------------
    # --- Write all sheets (including updated “sum” sheets) ---------
    # ----------------------------------------------------------------
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 1) ResultT – hourly
        df_T.to_excel(writer, sheet_name='ResultT_all', index=False)

        # 2) ResultTsum – pivoted & ordered
        df_Tsum.to_excel(writer, sheet_name='ResultTsum', index=False)

        # 3) Flows – hourly
        df_F.to_excel(writer, sheet_name='Flows', index=False)

        # 4) ResultFsum – pivoted
        df_Fsum = (
            df_F
              .set_index(['areaFrom','areaTo','energy'])[time_cols]
              .sum(axis=1)             # sum each (areaFrom, areaTo, energy) over all t
              .unstack(fill_value=0)   # pivot so “energy” becomes columns
              .reset_index()           # bring “areaFrom” & “areaTo” back as columns
        )
        df_Fsum.to_excel(writer, sheet_name='ResultFsum', index=False)

        # 5) ResultA – hourly
        df_A.to_excel(writer, sheet_name='ResultA_all', index=False)

        # 6) ResultAsum – pivoted, with “Electricity” prices averaged
        df_Asum.to_excel(writer, sheet_name='ResultAsum', index=False)

        # 7) ResultC – hourly capacity factors
        df_C_hourly.to_excel(writer, sheet_name='ResultC', index=False, startrow=0)
        #    blank row, then summary (CF_Summary & FLH)
        df_C_summary.to_excel(
            writer,
            sheet_name='ResultC',
            index=False,
            startrow=len(df_C_hourly) + 2
        )

        # 8) Duals – hourly and weekly dual values
        if model.Demand_Target:
            # write hourly CO2 at the top
            df_co2.to_excel(writer, sheet_name='Duals', index=False, startrow=0, startcol=0)
            # then leave one blank line and write the weekly table
            df_meth.to_excel(writer, sheet_name='Duals', index=False, startrow=0, startcol=3)

        #Objective function decomposition
        df_decomp.to_excel(writer, sheet_name="ObjDecomp", index=False)