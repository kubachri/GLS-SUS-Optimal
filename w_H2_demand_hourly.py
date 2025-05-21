#!/usr/bin/env python3
import os
import pandas as pd
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Constraint, Binary,
    NonNegativeReals, SolverFactory, Objective, maximize
)
import logging
from pyomo.util.infeasible import log_infeasible_constraints
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

    # 2.g) H₂ demand & sale price
    # fixed demand of 0.5 each hour
    demand_H2 = { t: 0.005 for t in T }
    # price from Price.inc: DK1.HydrogenCom.Export
    price_H2  = { t: data['price_sell'][('DK1','HydrogenCom', t)] for t in T }

    ################# DEBUG ####################
    print("Electrolyser Pmax * profile:",
      {t: Pmax['Electrolysis'] * profile['Electrolysis', t]
       for t in list(T)[:3]})  # first 3 hours

    print("Initial SOC:", soc_init)
    print("StorageCap:", soc_max)
    ################# DEBUG ####################

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

    model.h2_demand = Param(model.T,
                            initialize=demand_H2,
                            within=NonNegativeReals)
    model.h2_price  = Param(model.T,
                            initialize=price_H2,
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

    model.h2_short = Var(model.T, within=NonNegativeReals)

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

    # --- NEW: ChargingStorageMax: cap charging by nameplate power when in charge mode ---
    def charging_max(m, g, t):
        return m.FuelTotalSto[g, t] <= m.PMAX[g] * m.Mode[g, t]
    model.ChargingMax = Constraint(model.G_s, model.T, rule=charging_max)

    # --- NEW: DischargingStorageMax: cap discharge by nameplate power when discharging ---
    def discharging_max(m, g, t):
        return m.Discharge[g, t] <= m.PMAX[g] * (1 - m.Mode[g, t])
    model.DischargingMax = Constraint(model.G_s, model.T, rule=discharging_max)

    # --- NEW: TerminalSOC: force end‐of‐horizon SOC back to initial SOC ---
    # def terminal_soc(m, g):
    #     return m.Volume[g, m.T.last()] == m.soc_init[g]
    # model.TerminalSOC = Constraint(model.G_s, rule=terminal_soc)

    def h2_demand_soft(m, t):
        # collect any ‘Generation’ variables (in case you add non‐storage H2 techs later)
        prod_terms = [
            m.Generation[g, e, t]
            for (g, e) in m.OUT_p
            if e == 'HydrogenCom'
        ]
        # add storage discharge into HydrogenCom
        prod_terms += [
            m.GenerationSto[g, e, t]
            for (g, e) in m.STO_OUT
            if e == 'HydrogenCom'
        ]
        if not prod_terms:
            return Constraint.Skip
        # now include the slack
        return sum(prod_terms) + m.h2_short[t] >= m.h2_demand[t]

    model.H2Demand = Constraint(model.T, rule=h2_demand_soft)



    # 5) Encourage storage cycling
    penalty = 1e3  # or some suitably large number

    model.Profit = Objective(
        expr=(
            sum(
                model.h2_price[t]
                * sum(
                    model.GenerationSto[g, 'HydrogenCom', t]
                    for (g, e) in model.OUT_p
                    if e == 'HydrogenCom'
                )
                for t in model.T
            )
            - penalty * sum(model.h2_short[t] for t in model.T)
        ),
        sense=maximize
    )

    # 6) Solve and debug infeasibility
    solver = SolverFactory('gurobi')
    solver.options['TimeLimit'] = 300
    solver.options['MIPGap']    = 0.10

    # 6.a) Write out an LP with symbolic labels for external IIS analysis if needed
    model.write('model_debug.lp', io_options={'symbolic_solver_labels': True})

    # 6.b) Configure the Pyomo infeasibility logger
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('pyomo.util.infeasible').setLevel(logging.INFO)

    # 6.c) Solve once
    results = solver.solve(model, tee=True)

    # 6.d) If infeasible, print all violated constraints
    # log_infeasible_constraints(model)

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

    rows = []
    for t in model.T:
        # total H2 production this hour
        prod = sum(
            model.Generation[g, 'HydrogenCom', t].value
            for (g, e) in model.OUT_p
            if e == 'HydrogenCom'
        ) + sum(
            model.GenerationSto[g, 'HydrogenCom', t].value
            for (g, e) in model.STO_OUT
            if e == 'HydrogenCom'
        )
        slack = model.h2_short[t].value
        rows.append({'time': t, 'production': prod, 'slack': slack})

    df_debug = pd.DataFrame(rows).set_index('time')
    if df_debug.empty:
        print("All demands met.  No problem hours.")
    else:
        print(df_debug[df_debug['slack'] > 0])

    # get the hours where demand isn’t met
    hours = list(df_debug[df_debug['slack'] > 0].index)

    rows = []
    for t in hours:
        # 1) HydrogenStorage state‐of‐charge, charge and discharge
        soc       = model.Volume['HydrogenStorage',    t].value
        charge    = model.FuelTotalSto['HydrogenStorage', t].value
        discharge = model.Discharge['HydrogenStorage',   t].value

        # 2) Solar & Wind electricity production
        wind_prod  = sum(
            model.Generation[g, 'Electricity', t].value
            for (g,e) in model.OUT_p
            if e=='Electricity' and 'Wind'  in g
        )
        solar_prod = sum(
            model.Generation[g, 'Electricity', t].value
            for (g,e) in model.OUT_p
            if e=='Electricity' and 'Solar' in g
        )

        # 3) Electrolysis hydrogen production (before storage)
        elec_prod = sum(
            model.Generation[g, 'HydrogenCom', t].value
            for (g,e) in model.OUT_p
            if e=='HydrogenCom' and 'Electrolysis' in g
        )

        rows.append({
            'time':        t,
            'soc_H2':      soc,
            'charge_H2':   charge,
            'discharge_H2':discharge,
            'wind_el':     wind_prod,
            'solar_el':    solar_prod,
            'elec_H2_prod':elec_prod,
            'slack':       model.h2_short[t].value
        })

    if rows:
        df_metrics = pd.DataFrame(rows).set_index('time')
        print(df_metrics)
    else:
        print("✅ All H₂ demands met. No bottleneck hours.")

    # Check end‐of‐horizon SOC vs initial
    end_t = model.T.last()
    start_soc = model.soc_init['HydrogenStorage']
    end_soc   = model.Volume['HydrogenStorage', end_t].value
    print(f"Start SOC = {start_soc:.3f}, End SOC = {end_soc:.3f}")


    with pd.ExcelWriter('storage_with_market.xlsx', engine='xlsxwriter') as w:
        df_prod .to_excel(w, sheet_name='Production', index=False)
        df_store.to_excel(w, sheet_name='Storage',    index=False)

    print("✅ storage_with_market.xlsx written")

if __name__ == "__main__":
    main()