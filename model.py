#!/usr/bin/env python3
import os
import pandas as pd
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Constraint,
    NonNegativeReals, SolverFactory, Objective, maximize
)
from src.data_loader import load_data, load_techdata

def main():
    # 1) Load data
    INC_DIR  = os.path.join(os.path.dirname(__file__), "inc_data")
    data     = load_data()
    tech_df  = load_techdata()

    # 2) Extract sets & parameters
    G         = data['G']
    T         = data['T']
    sigma_in  = data['sigma_in']
    sigma_out = data['sigma_out']
    Pmax      = data.get('pmax', data['Pmax'])
    profile   = data['Profile']
    G_s       = data['G_s']              # storage technologies
    G_p       = [g for g in G if g not in G_s]  # producers

    # 2a) storage → its import carrier (prefer non-electricity)
    storage_fuel = {}
    for g in G_s:
        cands = [f for (gg,f),v in sigma_in.items() if gg==g and v>0]
        chems = [f for f in cands if f.lower() != "electricity"]
        storage_fuel[g] = chems[0] if chems else cands[0]

    # 2b) storage → its main export fuel (sigma_out == 1)
    main_fuel = {
        g: f
        for (g,f),v in sigma_out.items()
        if g in G_s and abs(v - 1.0) < 1e-6
    }

    # 2c) price: €2000/t only for HydrogenCom
    price = {g: (2000.0 if main_fuel[g] == "HydrogenCom" else 0.0)
             for g in G_s}

    # 2d) SOC init & cap from Techdata.inc
    soc_init = {g: tech_df.at[g, "InitialVolume"] for g in G_s}
    soc_max  = {g: tech_df.at[g, "StorageCap"]    for g in G_s}

    # 3) Build model
    model = ConcreteModel()

    # Sets
    model.G_p  = Set(initialize=G_p)
    model.G_s  = Set(initialize=G_s)
    model.G    = Set(initialize=G)
    model.T    = Set(initialize=T)
    model.IN_p = Set(dimen=2,
                     initialize=[(g,f) for (g,f),v in sigma_in.items()
                                 if v>0 and g in G_p])
    model.OUT_p= Set(dimen=2,
                     initialize=[(g,f) for (g,f),v in sigma_out.items()
                                 if v>0 and g in G_p])

    # Parameters
    model.sigma_in  = Param(model.IN_p,
                            initialize=lambda m,g,f: sigma_in[g,f],
                            within=NonNegativeReals)
    model.sigma_out = Param(model.OUT_p,
                            initialize=lambda m,g,f: sigma_out[g,f],
                            within=NonNegativeReals)
    model.PMAX      = Param(model.G,
                            initialize=lambda m,g: Pmax[g],
                            within=NonNegativeReals)
    model.Profile   = Param(model.G, model.T,
                            initialize=lambda m,g,t: profile[g,t],
                            within=NonNegativeReals)
    model.soc_init  = Param(model.G_s,
                            initialize=lambda m,g: soc_init[g],
                            within=NonNegativeReals)
    model.soc_max   = Param(model.G_s,
                            initialize=lambda m,g: soc_max[g],
                            within=NonNegativeReals)
    model.price     = Param(model.G_s,
                            initialize=lambda m,g: price[g],
                            within=NonNegativeReals)

    # Variables
    model.FuelTotal  = Var(model.G_p,       model.T, domain=NonNegativeReals)
    model.FuelUse    = Var(model.IN_p,      model.T, domain=NonNegativeReals)
    model.Generation = Var(model.OUT_p,     model.T, domain=NonNegativeReals)
    model.Charge     = Var(model.G_s,       model.T, domain=NonNegativeReals)
    model.Discharge  = Var(model.G_s,       model.T, domain=NonNegativeReals)
    model.Volume     = Var(model.G_s,       model.T, domain=NonNegativeReals)
    model.Sell       = Var(model.G_s,       model.T, domain=NonNegativeReals)

    # 4) Constraints

    # 4.1) producer fuel‐mix
    def prod_fmix(m, g, f, t):
        return model.FuelUse[g,f,t] == model.sigma_in[g,f] * model.FuelTotal[g,t]
    model.ProdFuelMix = Constraint(model.IN_p, model.T, rule=prod_fmix)

    # 4.2) producer output‐mix
    def prod_out(m, g, f, t):
        return model.Generation[g,f,t] == model.sigma_out[g,f] * model.FuelTotal[g,t]
    model.ProdOutMix = Constraint(model.OUT_p, model.T, rule=prod_out)

    # 4.3) cap producers by Pmax×profile
    def cap_prod(m, g, t):
        return model.FuelTotal[g,t] <= model.PMAX[g] * model.Profile[g,t]
    model.CapProd = Constraint(model.G_p, model.T, rule=cap_prod)

    # 4.4) fix producers at max output
    for g in G_p:
        for t in T:
            model.FuelTotal[g,t].fix(Pmax[g] * profile[g,t])

    # 4.5) SOC balance (charge & discharge)
    def soc_bal(m, g, t):
        prev = model.soc_init[g] if t==model.T.first() else model.Volume[g,model.T.prev(t)]
        return model.Volume[g,t] == prev + model.Charge[g,t] - model.Discharge[g,t]
    model.SOCBalance = Constraint(model.G_s, model.T, rule=soc_bal)

    # 4.6) SOC cap
    def soc_cap(m, g, t):
        return model.Volume[g,t] <= model.soc_max[g]
    model.SOCCap = Constraint(model.G_s, model.T, rule=soc_cap)

    # 4.7) charge ≤ actual upstream production of that carrier
    def charge_lim(m, g, t):
        f = storage_fuel[g]
        return model.Charge[g,t] <= sum(
            model.Generation[p,f,t] for (p,f0) in model.OUT_p if f0 == f
        )
    model.ChargeLim = Constraint(model.G_s, model.T, rule=charge_lim)

    # 4.8) discharge ≤ previous SOC
    def discharge_lim(m, g, t):
        prev = model.soc_init[g] if t==model.T.first() else model.Volume[g,model.T.prev(t)]
        return model.Discharge[g,t] <= prev
    model.DischargeLim = Constraint(model.G_s, model.T, rule=discharge_lim)

    # 4.9) sell ≤ discharge
    def sell_lim(m, g, t):
        return model.Sell[g,t] <= model.Discharge[g,t]
    model.SellLim = Constraint(model.G_s, model.T, rule=sell_lim)

    # 4.10) cycSOC: end = start
    model.CycSOC = Constraint(
        model.G_s,
        rule=lambda m,g: model.Volume[g,model.T.last()] == model.soc_init[g]
    )

    # 5) Objective: maximise H₂‐sales revenue
    model.Obj = Objective(
        expr=sum(model.Sell[g,t] * model.price[g] for g in model.G_s for t in model.T),
        sense=maximize
    )

    # 6) Solve
    SolverFactory('gurobi').solve(model, tee=True)

    # ─── DEBUG: show charge‐limit inputs ───
    print("\n=== DEBUG CHARGE LIMIT ===")
    for g in model.G_s:
        f_in = storage_fuel[g]
        print(f"\nStorage tech: {g} (import carrier = {f_in})")
        print("Hour   |   SumGen(producers→f_in)   |   Charge(var)")
        print("-----------------------------------------------")
        for t in model.T:
            # sum of all producer outputs of carrier f_in
            rhs = sum(
                model.Generation[p, f_in, t].value
                for (p, f0) in model.OUT_p
                if f0 == f_in
            )
            lhs = model.Charge[g, t].value
            print(f"{t:7s} | {rhs:24.6f} | {lhs:12.6f}")
    print("=== END DEBUG ===\n")

    # 7) Export to Excel

    # Production sheet
    prod_rows = []
    for (g,f) in model.OUT_p:
        for t in model.T:
            prod_rows.append({
                'tech':    g,
                'carrier': f,
                'time':    t,
                'output':  model.Generation[g,f,t].value
            })
    df_prod = pd.DataFrame(prod_rows)

    # Storage sheet
    store_rows = []
    for g in model.G_s:
        for t in model.T:
            store_rows.append({
                'tech':      g,
                'time':      t,
                'charge':    model.Charge[g,t].value,
                'discharge': model.Discharge[g,t].value,
                'soc':       model.Volume[g,t].value,
                'sell':      model.Sell[g,t].value
            })
    df_store = pd.DataFrame(store_rows)

    with pd.ExcelWriter('storage_with_market.xlsx', engine='xlsxwriter') as w:
        df_prod .to_excel(w, sheet_name='Production', index=False)
        df_store.to_excel(w, sheet_name='Storage',    index=False)

    print("✅ storage_with_market.xlsx written")

if __name__ == "__main__":
    main()
