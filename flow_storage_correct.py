#!/usr/bin/env python3
import os
import pandas as pd
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Constraint, Binary,
    NonNegativeReals, SolverFactory, Objective, maximize
)
from src.data_loader import load_data, load_techdata

def main():
    # 1) Load data
    data    = load_data()
    tech_df = load_techdata()

    # 2) Extract sets & raw parameters
    G         = data['G']
    T         = data['T']
    sigma_in  = data['sigma_in'].copy()
    sigma_out = data['sigma_out'].copy()
    Pmax      = data.get('pmax', data['Pmax']).copy()
    profile   = data['Profile']
    G_s       = data['G_s']
    G_p       = [g for g in G if g not in G_s]

    # 2.a) All energy carriers, sorted
    carriers = sorted({f for (g,f) in sigma_in} | {f for (g,f) in sigma_out})

    # ─── 2.1) UC / RR / scale capacity ─────────────────────────────────
    orig_cap   = tech_df['Capacity'].copy()
    orig_min   = tech_df['Minimum'].copy()
    orig_ramp  = tech_df['RampRate'].copy()
    sum_in_raw = {
        g: sum(v for (gg,e),v in sigma_in.items() if gg==g)
        for g in tech_df.index
    }

    UC = [g for g in tech_df.index if orig_min[g]  > 0]
    RR = [g for g in tech_df.index if orig_ramp[g] > 0]

    for g in UC:
        tech_df.at[g,'Minimum'] = sum_in_raw[g]*orig_cap[g]*orig_min[g]
    for g in RR:
        tech_df.at[g,'RampRate'] = sum_in_raw[g]*orig_cap[g]*orig_ramp[g]
    for g in tech_df.index:
        newc = sum_in_raw[g] * orig_cap[g]
        tech_df.at[g,'Capacity'] = newc
        Pmax[g] = newc
    # ─────────────────────────────────────────────────────────────────

    # 2.b) Fe for every technology
    Fe = {}
    for g in tech_df.index:
        tot_in  = sum(v for (gg,e),v in sigma_in.items()  if gg==g)
        tot_out = sum(v for (gg,e),v in sigma_out.items() if gg==g)
        Fe[g]   = (tot_out/tot_in) if tot_in>0 else 1.0

    # 2.c) normalize raw mixes → fractions
    in_frac, out_frac = {}, {}
    for g in tech_df.index:
        sin  = sum(v for (gg,e),v in sigma_in.items()  if gg==g)
        sout = sum(v for (gg,e),v in sigma_out.items() if gg==g)
        for (gg,e),v in sigma_in.items():
            if gg==g:
                in_frac[(g,e)] = v/sin  if sin>0  else 0
        for (gg,e),v in sigma_out.items():
            if gg==g:
                out_frac[(g,e)] = v/sout if sout>0 else 0

    # 2.d) storage import‐carrier
    storage_fuel = {}
    for g in G_s:
        cands = [f for (gg,f),v in sigma_in.items() if gg==g and v>0]
        chems = [f for f in cands if f.lower()!='electricity']
        storage_fuel[g] = chems[0] if chems else cands[0]

    # 2.e) SOC init & max
    soc_init = {g: tech_df.at[g,'InitialVolume'] for g in G_s}
    soc_max  = {g: tech_df.at[g,'StorageCap']    for g in G_s}

    # 2.f) big‐M for storage
    bigM_charge = {
        g: sum(Pmax[p]*sigma_out.get((p,storage_fuel[g]),0) for p in G_p)
        for g in G_s
    }
    bigM_disc = {g: soc_max[g] for g in G_s}

    # 3) Build model
    model = ConcreteModel()

    # --- Sets ---
    model.G       = Set(initialize=G)
    model.E       = Set(initialize=carriers)
    model.G_p     = Set(initialize=G_p)
    model.G_s     = Set(initialize=G_s)
    model.T       = Set(initialize=T)
    model.IN_p    = Set(dimen=2, initialize=[(g,f) for (g,f),v in sigma_in.items()
                                              if v>0 and g in G_p])
    model.OUT_p   = Set(dimen=2, initialize=[(g,f) for (g,f),v in sigma_out.items()
                                              if v>0 and g in G_p])
    model.STO_IN  = Set(dimen=2, initialize=[(g,f) for (g,f),v in sigma_in.items()
                                              if v>0 and g in G_s])
    model.STO_OUT = Set(dimen=2, initialize=[(g,f) for (g,f),v in sigma_out.items()
                                              if v>0 and g in G_s])

    # --- Parameters ---
    model.PMAX     = Param(model.G,          initialize=Pmax,       within=NonNegativeReals)
    model.Profile  = Param(model.G, model.T, initialize=profile,    within=NonNegativeReals)
    model.Fe       = Param(model.G,          initialize=Fe,         within=NonNegativeReals)
    model.soc_init = Param(model.G_s,        initialize=soc_init,   within=NonNegativeReals)
    model.soc_max  = Param(model.G_s,        initialize=soc_max,    within=NonNegativeReals)
    model.BigM_C   = Param(model.G_s,        initialize=bigM_charge,within=NonNegativeReals)
    model.BigM_D   = Param(model.G_s,        initialize=bigM_disc,  within=NonNegativeReals)

    model.in_frac  = Param(model.G, model.E,
                           initialize=lambda m,g,e: in_frac.get((g,e),0),
                           within=NonNegativeReals)
    model.out_frac = Param(model.G, model.E,
                           initialize=lambda m,g,e: out_frac.get((g,e),0),
                           within=NonNegativeReals)

    # --- Variables ---
    model.FuelTotal     = Var(model.G_p,   model.T,      within=NonNegativeReals)
    model.FuelUse       = Var(model.IN_p,  model.T,      within=NonNegativeReals)
    model.Generation    = Var(model.OUT_p, model.T,      within=NonNegativeReals)

    # storage
    model.FuelTotalSto  = Var(model.G_s,   model.T,      within=NonNegativeReals)
    model.FuelUseSto    = Var(model.STO_IN, model.T,      within=NonNegativeReals)
    model.Discharge     = Var(model.G_s,   model.T,      within=NonNegativeReals)
    model.GenerationSto = Var(model.STO_OUT,model.T,      within=NonNegativeReals)
    model.Volume        = Var(model.G_s,   model.T,      within=NonNegativeReals)
    model.Sell          = Var(model.G_s,   model.T,      within=NonNegativeReals)
    model.Mode          = Var(model.G_s,   model.T,      domain=Binary)

    # --- Constraints ---

    # 4.1) prod fuel‐mix
    def prod_fuelmix(m,g,f,t):
        return model.FuelUse[g,f,t] == model.in_frac[g,f] * model.FuelTotal[g,t]
    model.ProdFuelMix = Constraint(model.IN_p, model.T, rule=prod_fuelmix)

    # 4.2) prod output
    def prod_out(m,g,f,t):
        return model.Generation[g,f,t] == model.out_frac[g,f] * model.FuelTotal[g,t] * model.Fe[g]
    model.ProdOut = Constraint(model.OUT_p, model.T, rule=prod_out)

    # 4.3) capacity limit
    def cap_prod(m,g,t):
        return model.FuelTotal[g,t] <= model.PMAX[g] * model.Profile[g,t]
    model.CapProd = Constraint(model.G_p, model.T, rule=cap_prod)

    # debug: fix at max
    for g in G_p:
        for t in T:
            model.FuelTotal[g,t].fix(Pmax[g] * profile[g,t])

    # 4.5) storage fuel‐mix
    def sto_fuelmix(m,g,f,t):
        return model.FuelUseSto[g,f,t] == model.in_frac[g,f] * model.FuelTotalSto[g,t]
    model.StorageFuelMix = Constraint(model.STO_IN, model.T, rule=sto_fuelmix)

    # 4.6) storage output (no Fe on discharge)
    def sto_prod(m,g,f,t):
        return m.GenerationSto[g,f,t] == m.out_frac[g,f] * m.Discharge[g,t]
    model.StorageProd = Constraint(model.STO_OUT, model.T, rule=sto_prod)

    # 4.7) GAMS ProductionStorage
    def storage_balance(m,g,t):
        prev = model.soc_init[g] if t==model.T.first() else model.Volume[g,model.T.prev(t)]
        out_flow = sum(
            model.GenerationSto[g,e,t]
            for (gg,e) in model.STO_OUT
            if gg==g
        )
        return model.Volume[g,t] == prev \
               + model.FuelTotalSto[g,t]*model.Fe[g] \
               - out_flow
    model.StorageBalance = Constraint(model.G_s, model.T, rule=storage_balance)

    # 4.8) SOC cap
    def soc_cap(m,g,t):
        return model.Volume[g,t] <= model.soc_max[g]
    model.SOCCap = Constraint(model.G_s, model.T, rule=soc_cap)

    # 4.9) charge ≤ upstream hydrogen (or chosen) leg
    def charge_lim(m,g,t):
        f = storage_fuel[g]
        return model.FuelUseSto[g,f,t] <= sum(
            model.Generation[p,f,t] for (p,f0) in model.OUT_p if f0==f
        )
    model.ChargeLim = Constraint(model.G_s, model.T, rule=charge_lim)

    # 4.10) discharge ≤ previous SOC
    def discharge_lim(m,g,t):
        prev = model.soc_init[g] if t==model.T.first() else model.Volume[g,model.T.prev(t)]
        return model.Discharge[g,t] <= prev
    model.DischargeLim = Constraint(model.G_s, model.T, rule=discharge_lim)

    # 4.11) mode‐switch big‐M
    model.ChargeMode    = Constraint(model.G_s, model.T,
                             rule=lambda m,g,t: model.FuelTotalSto[g,t] <= model.BigM_C[g]*model.Mode[g,t])
    model.DischargeMode = Constraint(model.G_s, model.T,
                             rule=lambda m,g,t: model.Discharge[g,t]  <= model.BigM_D[g]*(1-model.Mode[g,t]))

    # 5) Encourage storage cycling
    model.Obj = Objective(
        expr=sum(model.FuelTotalSto[g,t] + model.Discharge[g,t]
                 for g in model.G_s for t in model.T),
        sense=maximize
    )

    # 6) Solve (30s feasibility)
    solver = SolverFactory('gurobi')
    solver.options['TimeLimit'] = 30
    solver.options['MIPGap']    = 0.10
    solver.solve(model, tee=True)

    # 7) Export
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

    store_rows = []
    for g in model.G_s:
        for t in model.T:
            row = {
                'tech':         g,
                'time':         t,
                'FuelTotalSto': model.FuelTotalSto[g,t].value,
                'discharge':    model.Discharge[g,t].value,
                'soc':          model.Volume[g,t].value
            }
            for (gg,f) in model.STO_IN:
                if gg==g:
                    row[f'in_{f}'] = model.FuelUseSto[g,f,t].value
            for (gg,f) in model.STO_OUT:
                if gg==g:
                    row[f'out_{f}'] = model.GenerationSto[g,f,t].value
            store_rows.append(row)
    df_store = pd.DataFrame(store_rows)

    with pd.ExcelWriter('storage_with_market.xlsx', engine='xlsxwriter') as w:
        df_prod .to_excel(w, sheet_name='Production', index=False)
        df_store.to_excel(w, sheet_name='Storage',    index=False)

    print("✅ storage_with_market.xlsx written")

if __name__ == "__main__":
    main()
