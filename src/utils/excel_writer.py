import pandas as pd
from pyomo.environ import value
from pathlib import Path

def export_results_to_excel(model, path="results.xlsx"):
    """
    Export solved Pyomo model results to an Excel file under `results/results.xlsx`
    (relative to the project root), creating the folder if it doesn’t exist.

    Parameters
    ----------
    model : ConcreteModel
        A solved Pyomo model containing T, f_in, f_out, G_s, Volume, Buy, Sale, etc.
    path : str, optional
        Explicit path to the output .xlsx file. If None, defaults to
        '<project_root>/results/results.xlsx'.
    """

    # Determine output path
    if path is None:
        # assume this file lives in <project_root>/src/utils/excel_writer.py
        project_root = Path(__file__).parents[2]
        output_path = project_root / "results" / "results.xlsx"
    else:
        output_path = Path(path)

    # Make sure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Time periods
    times = list(model.T)
    time_cols = [str(t) for t in times]

    # Helper to build DataFrame
    def make_df(rows, cols):
        return pd.DataFrame(rows, columns=cols)

    # Derive storage_fuel from model.f_in / model.f_out
    storage_fuel = {}
    for g in model.G_s:
        # model.f_in has exactly the (tech,fuel) pairs with sigma_in>0
        fuels_in = [f for (gg, f) in model.f_in if gg == g]
        if fuels_in:
            storage_fuel[g] = fuels_in[0]
        else:
            # fall back to exports if no import mix
            fuels_out = [f for (gg, f) in model.f_out if gg == g]
            storage_fuel[g] = fuels_out[0] if fuels_out else None

    # 1) Fuel use sheet
    fu_rows = []
    for g, e in model.f_in:
        row = {'Tech': g, 'Fuel': e}
        for t in times:
            row[str(t)] = value(model.Fueluse[g, e, t])
        fu_rows.append(row)
    df_fu = make_df(fu_rows, ['Tech', 'Fuel'] + time_cols)

    # 2) Generation sheet
    gen_rows = []
    for g, e in model.f_out:
        row = {'Tech': g, 'Fuel': e}
        for t in times:
            row[str(t)] = value(model.Generation[g, e, t])
        gen_rows.append(row)
    df_gen = make_df(gen_rows, ['Tech', 'Fuel'] + time_cols)

    # 3) Storage SOC sheet
    st_rows = []
    for g in model.G_s:
        row = {
            'Tech': g,
            'Fuel': storage_fuel[g],  # now always defined
            **{str(t): value(model.Volume[g,t]) for t in model.T}
        }
        st_rows.append(row)
    df_st = pd.DataFrame(st_rows, columns=['Tech', 'Fuel'] + time_cols)

    # 4) Imports (buys) sheet
    imp_rows = []
    for a, e in model.buyE:
        row = {'Area': a, 'Energy': e}
        for t in times:
            row[str(t)] = value(model.Buy[a, e, t])
        imp_rows.append(row)
    df_imp = make_df(imp_rows, ['Area', 'Energy'] + time_cols)

    # 5) Exports (sales) sheet
    exp_rows = []
    for a, e in model.saleE:
        row = {'Area': a, 'Energy': e}
        for t in times:
            row[str(t)] = value(model.Sale[a, e, t])
        exp_rows.append(row)
    df_exp = make_df(exp_rows, ['Area', 'Energy'] + time_cols)

    # Write to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_fu.to_excel(writer, sheet_name='FuelUse', index=False)
        df_gen.to_excel(writer, sheet_name='Generation', index=False)
        df_st.to_excel(writer, sheet_name='Storage', index=False)
        df_imp.to_excel(writer, sheet_name='Imports', index=False)
        df_exp.to_excel(writer, sheet_name='Exports', index=False)

    print(f"Wrote model results to {output_path.resolve()}")