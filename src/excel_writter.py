import pandas as pd
from pyomo.environ import SolverFactory, value
from model_final import main        # or whatever module you built your model in
from src.data_loader import load_data

def export_results_to_excel(path="results.xlsx"):
    # 1) build & solve
    m = main()
    SolverFactory('gurobi').solve(m)
    data = load_data()
    storage_fuel = data['storage_fuel']
    times = list(m.T)
    time_cols = [str(t) for t in times]

    # helper to turn a list of rows into a DataFrame with a given header order
    def make_df(rows, cols):
        return pd.DataFrame(rows, columns=cols)

    # 2) Fuel‐use sheet
    fu_rows = []
    for g,e in m.f_in:
        row = {'Tech': g, 'Fuel': e}
        for t in times:
            row[str(t)] = value(m.Fueluse[g,e,t])
        fu_rows.append(row)
    df_fu = make_df(fu_rows, ['Tech','Fuel']+time_cols)

    # 3) Generation sheet
    gen_rows = []
    for g,e in m.f_out:
        row = {'Tech': g, 'Fuel': e}
        for t in times:
            row[str(t)] = value(m.Generation[g,e,t])
        gen_rows.append(row)
    df_gen = make_df(gen_rows, ['Tech','Fuel']+time_cols)

    # 4) Storage SOC sheet
    st_rows = []
    for g in m.G_s:
        row = {'Tech': g, 'Fuel': storage_fuel[g]}
        for t in times:
            row[str(t)] = value(m.Volume[g,t])
        st_rows.append(row)
    df_st = make_df(st_rows, ['Tech','Fuel']+time_cols)

    # 5) Imports (buys)
    imp_rows = []
    for a,e in m.buyE:
        row = {'Area': a, 'Energy': e}
        for t in times:
            row[str(t)] = value(m.Buy[a,e,t])
        imp_rows.append(row)
    df_imp = make_df(imp_rows, ['Area','Energy']+time_cols)

    # 6) Exports (sales)
    exp_rows = []
    for a,e in m.saleE:
        row = {'Area': a, 'Energy': e}
        for t in times:
            row[str(t)] = value(m.Sale[a,e,t])
        exp_rows.append(row)
    df_exp = make_df(exp_rows, ['Area','Energy']+time_cols)

    # 7) Write out to Excel
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        df_fu.to_excel(writer, sheet_name='FuelUse', index=False)
        df_gen.to_excel(writer, sheet_name='Generation', index=False)
        df_st.to_excel(writer, sheet_name='Storage', index=False)
        df_imp.to_excel(writer, sheet_name='Imports', index=False)
        df_exp.to_excel(writer, sheet_name='Exports', index=False)

    print(f"Wrote all results to {path}")