#!/usr/bin/env python3
import os
import pandas as pd
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Constraint, Binary, Reals,
    NonNegativeReals, SolverFactory, Objective, maximize, value
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

    model.A      = Set(initialize=data['A'])
    area_of_tech = data['area_of']   # tech → area dict

    # 2) Flow‐set (area_in, area_out, energy)
    model.FlowSet = Set(dimen=3, initialize=data['FlowSet'])

    # 3) Which commodities can be bought/sold?
    #    (any (area,energy) pair that appears in the price dicts)
    buy_pairs  = sorted({(a,e) for (a,e,t) in data['price_buy']})
    sell_pairs = sorted({(a,e) for (a,e,t) in data['price_sell']})
    model.BuyE  = Set(dimen=2, initialize=buy_pairs)
    model.SaleE = Set(dimen=2, initialize=sell_pairs)

    # extract numeric hour
    hour_nums = {t: int(t.split('-',1)[1]) for t in T}
    week_of    = {t: ((h-1)//168) + 1 for t,h in hour_nums.items()}
    weeks      = sorted(set(week_of.values()))
    # for each week, pick the hour with the highest number
    hours_by_week = {}
    for t,w in week_of.items():
        hours_by_week.setdefault(w, []).append(t)
    last_hour = { w: max(lst, key=lambda t: hour_nums[t])
                  for w,lst in hours_by_week.items() }

    # ─── add to Pyomo ───
    model.W = Set(initialize=weeks)

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

    # weekly delivery requirement: 2 tons/week
    weekly_demand = {w: 2.0 for w in weeks}
    model.weekly_demand = Param(model.W,
                                initialize=weekly_demand,
                                within=NonNegativeReals)

    # commodity prices from data_loader
    model.price_buy  = Param(model.A, model.E, model.T,
        initialize=lambda m,a,e,t: data['price_buy'].get((a,e,t), 0.0),
        within=Reals)
    model.price_sell = Param(model.A, model.E, model.T,
        initialize=lambda m,a,e,t: data['price_sell'].get((a,e,t), 0.0),
        within=Reals)

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
    model.Mode          = Var(model.G_s,   model.T,      domain=Binary)

    model.weekly_short = Var(model.W, within=NonNegativeReals)

    # decision to buy / sell on the DA market
    model.BuyCom  = Var(model.BuyE,  model.T, within=NonNegativeReals)
    model.SellCom = Var(model.SaleE, model.T, within=NonNegativeReals)
    model.Flow    = Var(model.FlowSet, model.T, within=NonNegativeReals)

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
    def terminal_soc(m, g):
        return m.Volume[g, m.T.last()] == m.soc_init[g]
    model.TerminalSOC = Constraint(model.G_s, rule=terminal_soc)

    def weekly_end_discharge_soft(m, w):
        t_end = last_hour[w]
        # only storage‐discharge variables at t_end count
        terms = [
            m.GenerationSto[g, 'HydrogenCom', t_end]
            for (g,e) in m.STO_OUT
            if e == 'HydrogenCom'
        ]
        if not terms:
            return Constraint.Skip
        # allow slack if you can’t meet 2.0 exactly
        return sum(terms) + m.weekly_short[w] >= m.weekly_demand[w]

    model.WeeklyEndDischarge = Constraint(model.W, rule=weekly_end_discharge_soft)

    def hub_balance(m, a, e, t):
        # 1) do we even need a constraint here?
        has_market = (a,e) in m.BuyE or (a,e) in m.SaleE
        has_tech   = any(
            area_of_tech[g]==a and ((g,e) in m.IN_p or (g,e) in m.OUT_p)
            for g in m.G
        )
        has_flow = any(
            (aa,a,e) in m.FlowSet or (a,aa,e) in m.FlowSet
            for aa in m.A
        )
        if not (has_market or has_tech or has_flow):
            return Constraint.Skip

        # 2) collect buys/sells
        buy  = m.BuyCom[a,e,t] if (a,e) in m.BuyE  else 0
        sell = m.SellCom[a,e,t] if (a,e) in m.SaleE else 0

        # 3) flows
        imports = sum(m.Flow[a0,a,e,t]
                      for (a0,aa,ee) in m.FlowSet
                      if aa==a and ee==e)
        exports = sum(m.Flow[a,a1,e,t]
                      for (aa,a1,ee) in m.FlowSet
                      if aa==a and ee==e)

        # 4) local prod/use
        prod_nonsto = sum(m.Generation[g,e,t]
                          for (g,ee) in m.OUT_p
                          if ee==e and area_of_tech[g]==a)
        prod_sto    = sum(m.GenerationSto[g,e,t]
                          for (g,ee) in m.STO_OUT
                          if ee==e and area_of_tech[g]==a)
        use_nonsto  = sum(m.FuelUse[g,e,t]
                          for (g,ee) in m.IN_p
                          if ee==e and area_of_tech[g]==a)
        use_sto     = sum(m.FuelUseSto[g,e,t]
                          for (g,ee) in m.STO_IN
                          if ee==e and area_of_tech[g]==a)

        # 5) enforce equality
        return (buy + imports + prod_nonsto + prod_sto
               ) == (use_nonsto + use_sto + sell + exports)

    model.HubBalance = Constraint(model.A, model.E, model.T, rule=hub_balance)

    # 5) Encourage storage cycling
    penalty = 1e3  # or some suitably large number

    penalty_weekly = 1e3

    def profit_expr(m):
        revenue = sum(
            m.price_sell[a,e,t] * m.SellCom[a,e,t]
            for (a,e) in m.SaleE for t in m.T
        )
        cost = sum(
            m.price_buy[a,e,t] * m.BuyCom[a,e,t]
            for (a,e) in m.BuyE for t in m.T
        )
        slack_penalty = penalty_weekly * sum(m.weekly_short[w] for w in m.W)
        return revenue - cost - slack_penalty

    model.Profit = Objective(rule=profit_expr, sense=maximize)



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

    # ─── DEBUG: check the balance for the first 3 hours ────────────────────
    print("\n--- Hub balances (first 3 hours) ---")
    for a in model.A:
        for e in ['Electricity','HydrogenCom']:
            for t in list(model.T)[:3]:
                lhs_expr = (
                    (model.BuyCom[a,e,t].value if (a,e) in model.BuyE else 0)
                    + sum(value(model.Flow[a0,a,e,t])
                          for (a0,aa,ee) in model.FlowSet
                          if aa==a and ee==e)
                    + sum(value(model.Generation[g,e,t])
                          for (g,ee) in model.OUT_p
                          if ee==e and area_of_tech[g]==a)
                    + sum(value(model.GenerationSto[g,e,t])
                          for (g,ee) in model.STO_OUT
                          if ee==e and area_of_tech[g]==a)
                )
                rhs_expr = (
                    sum(value(model.FuelUse[g,e,t])
                        for (g,ee) in model.IN_p
                        if ee==e and area_of_tech[g]==a)
                    + sum(value(model.FuelUseSto[g,e,t])
                          for (g,ee) in model.STO_IN
                          if ee==e and area_of_tech[g]==a)
                    + (model.SellCom[a,e,t].value if (a,e) in model.SaleE else 0)
                    + sum(value(model.Flow[a,a1,e,t])
                          for (aa,a1,ee) in model.FlowSet
                          if aa==a and ee==e)
                )
                print(f"{a:5s} | {e:12s} | {t:8s} | "
                      f"LHS={lhs_expr:.6f} | RHS={rhs_expr:.6f}")
    print("--- end debug ---\n")



    # 6.d) If infeasible, print all violated constraints
    # log_infeasible_constraints(model)

    # 7) Build the production DataFrame
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

    # 8) Build the storage DataFrame
    store_rows = []
    for g in model.G_s:
        for t in model.T:
            row = {
                'tech':       g,
                'time':       t,
                'FuelTotalSto': model.FuelTotalSto[g,t].value,
                'discharge':    model.Discharge[g,t].value,
                'soc':          model.Volume[g,t].value
            }
            for (gg,f) in model.STO_IN:
                if gg == g:
                    row[f'in_{f}'] = model.FuelUseSto[g,f,t].value
            for (gg,f) in model.STO_OUT:
                if gg == g:
                    row[f'out_{f}'] = model.GenerationSto[g,f,t].value
            store_rows.append(row)
    df_store = pd.DataFrame(store_rows)

    # 9) Build the production DataFrame
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

    # 10) Build the storage DataFrame
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

    # 11) Build the weekly‐contract debug table
    weekly_rows = []
    for w in model.W:
        t_end = last_hour[w]
        discharged = sum(
            model.GenerationSto[g,'HydrogenCom',t_end].value
            for (g,e) in model.STO_OUT if e=='HydrogenCom'
        )
        slack = model.weekly_short[w].value
        weekly_rows.append({
            'week':        w,
            'week_end':    t_end,
            'discharged':  discharged,
            'contract':    model.weekly_demand[w],
            'shortfall':   slack
        })

    df_weekly = pd.DataFrame(weekly_rows).set_index('week')
    print("\nWeekly delivery summary (per contract):")
    print(df_weekly)

    # 12) Check terminal SOC
    end_t     = model.T.last()
    start_soc = model.soc_init['HydrogenStorage']
    end_soc   = model.Volume['HydrogenStorage', end_t].value
    print(f"\nStart SOC = {start_soc:.3f}, End SOC = {end_soc:.3f}")

    # 13) Write everything to Excel
    with pd.ExcelWriter('storage_with_market.xlsx', engine='xlsxwriter') as w:
        df_prod.to_excel(w, sheet_name='Production', index=False)
        df_store.to_excel(w, sheet_name='Storage',    index=False)
        df_weekly.to_excel(w, sheet_name='WeeklyDelivery')
    print("✅ storage_with_market.xlsx written")

if __name__ == "__main__":
    main()
