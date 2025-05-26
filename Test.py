#!/usr/bin/env python3
import os
import pandas as pd
from pygments.lexer import default
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Constraint, Binary,
    NonNegativeReals, SolverFactory, Objective, maximize,
    Reals
)
from collections import defaultdict

from src.data_loader import load_data, load_techdata

def main():
    # 1) Load data
    data    = load_data()
    tech_df = load_techdata()

    # 2) Extract sets & raw parameters
    G           = data['G']
    T           = data['T']
    sigma_in    = data['sigma_in'].copy()
    sigma_out   = data['sigma_out'].copy()
    capacity    = data.get('capacity', data['capacity']).copy()
    profile     = data['Profile']
    G_s         = data['G_s']
    G_p         = [g for g in G if g not in G_s]
    location    = data['location']
    F           = data['F']
    flowset     = data['FlowSet']
    A           = data['A']
    Xcap        = data['Xcap']
    price_buy   = data['price_buy']
    price_sell  = data["price_sell"]
    cvar        = data['Cvar']
    cstart      = data['Cstart']
    demand      = data['Demand']

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
        capacity[g] = newc
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
        # collect only the non‐zero imports for this tech
        in_items = [((gg, e), v) for (gg, e), v in sigma_in.items() if gg == g and v > 0]
        total_in = sum(v for (_, _), v in in_items)
        if total_in > 0:
            for (_, e), v in in_items:
                in_frac[(g, e)] = v / total_in

        # same for exports
        out_items = [((gg, e), v) for (gg, e), v in sigma_out.items() if gg == g and v > 0]
        total_out = sum(v for (_, _), v in out_items)
        if total_out > 0:
            for (_, e), v in out_items:
                out_frac[(g, e)] = v / total_out

    soc_init = {g: tech_df.at[g, 'InitialVolume'] for g in G_s}
    soc_max = {g: tech_df.at[g, 'StorageCap'] for g in G_s}
    RampRate = tech_df['RampRate']
    Minimum = tech_df['Minimum']

    pairs_TechToEnergy = [
        (g, f)
        for g in G
        for f in F
        if data['sigma_out'][(g,f)] == 1
    ]

    pairs_out = [
        (g, f)
        for g in G
        for f in F
        if sigma_out.get((g, f), 0.0) > 0.0
    ]

    pairs_in = [
        (g, f)
        for g in G
        for f in F
        if sigma_in.get((g, f), 0.0) > 0.0
    ]

    pairs_buy = sorted({
    (area, energy)
    for (area, energy, t), price in price_buy.items()
    if price > 0.0
    })

    pairs_sale = sorted({
    (area, energy)
    for (area, energy, t), price in price_sell.items()
    if price > 0.0
    })

    area_has = defaultdict(bool)
    for (a, e, t), cap in Xcap.items():
        if cap > 0.0:
            area_has[a] = True

    areas_with_lines = [a for a, has in area_has.items() if has]

    print(tech_df)

     # 3) Build model
    model = ConcreteModel()

    # --- Sets ---
    model.A = Set(initialize=A)
    model.G = Set(initialize=G)
    model.F = Set(initialize=F)
    model.T = Set(initialize=T, ordered=True)
    model.G_s = Set(initialize=G_s, within=model.G)
    model.G_p = Set(initialize=G_p, within=model.G)
    model.flowset = Set(initialize=flowset, within=model.A * model.A * model.F, dimen=3)
    #'Link the technology with the fuel/energy which was designed and constraint e.g., electrolyzer to hydrogen, Methanol plant to methanol etc'
    model.TechToEnergy = Set(initialize=pairs_TechToEnergy, within=model.G * model.F, dimen=2)
    model.f_out = Set(initialize=pairs_out, within=model.G * model.F, dimen=2)
    model.f_in = Set(initialize=pairs_in, within=model.G * model.F, dimen=2)
    model.buyE = Set(initialize=pairs_buy, within=model.A * model.F, dimen=2)
    model.saleE = Set(initialize=pairs_sale, within=model.A * model.F, dimen=2)
    model.UC = Set(initialize=UC, within=model.G)
    model.RR = Set(initialize=RR, within=model.G)
    model.location = Set(initialize=location, within=model.A * model.G, dimen=2)
    model.LinesInterconnectors = Set(initialize=areas_with_lines, within=model.A)

    model.T.pprint()

    # --- Parameters ---
    model.Profile = Param(model.G, model.T, initialize=profile, within=NonNegativeReals)
    model.capacity = Param(model.G, initialize=capacity, within=NonNegativeReals)
    model.Fe       = Param(model.G,          initialize=Fe,         within=NonNegativeReals)
    model.soc_init = Param(model.G_s,        initialize=soc_init,   within=NonNegativeReals)
    model.soc_max  = Param(model.G_s,        initialize=soc_max,    within=NonNegativeReals)
    model.cstart   = Param(model.G, initialize=cstart, within=NonNegativeReals)
    model.cvar = Param(model.G, initialize=cvar, within=NonNegativeReals)
    model.RampRate = Param(model.G, initialize=RampRate, within=NonNegativeReals)
    model.Minimum = Param(model.G, initialize=Minimum, within=NonNegativeReals)
    model.in_frac  = Param(model.G, model.F, initialize=in_frac, within=NonNegativeReals)
    model.out_frac = Param(model.G, model.F, initialize=out_frac, within=NonNegativeReals)
    model.demand = Param(model.A, model.F, model.T, initialize=demand, within=NonNegativeReals)
    model.price_buy = Param(model.A, model.F, model.T, initialize=price_buy, within=Reals)
    model.price_sale = Param(model.A, model.F, model.T, initialize=price_sell, within=Reals)
    model.InterconnectorCapacity = Param(model.LinesInterconnectors, model.F, model.T,
                                         initialize=Xcap, default= 0, within=NonNegativeReals)

   # --- Variables ---
    model.Cost = Var(domain=NonNegativeReals)
    model.Fueluse = Var(model.f_in, model.T, domain=NonNegativeReals)
    model.Fuelusetotal = Var(model.G, model.T, domain=NonNegativeReals)
    model.Generation = Var(model.f_out, model.T, domain=NonNegativeReals)
    model.Volume = Var(model.G_s, model.T, domain=NonNegativeReals)
    model.Flow = Var(model.flowset, model.T, domain=NonNegativeReals)
    model.Buy = Var(model.buyE, model.T, domain=NonNegativeReals)
    model.Sale = Var(model.saleE, model.T, domain=NonNegativeReals)
    model.Startcost = Var(model.G, model.T, domain=NonNegativeReals)
    model.SlackDemandImport = Var(model.buyE, model.T, domain=NonNegativeReals)
    model.SlackDemandExport = Var(model.saleE, model.T, domain=NonNegativeReals)
    model.Online = Var(model.G, model.T, domain=Binary)
    model.Charge = Var(model.G_s, model.T, domain=Binary)

    # --- Constraints ---

    # 1) Flows imported to technologies
    def fuelmix_rule(m, g, e, t):
        if m.capacity[g] <= 0:
            return Constraint.Skip
        return m.in_frac[g,e] * m.Fuelusetotal[g,t] == m.Fueluse[g,e,t]
    model.Fuelmix = Constraint(model.f_in, model.T, rule=fuelmix_rule)

    # 2) Production for each non-storage technology
    def production_rule(m, g, e, t):
        if m.capacity[g] <= 0:
            return Constraint.Skip
        if g in m.G_s:
            return Constraint.Skip
        return m.out_frac[g,e] * m.Fuelusetotal[g,t] * m.Fe[g] == m.Generation[g,e,t]
    model.Production = Constraint(model.f_out, model.T, rule=production_rule)

    # 3) Storage constraints

    def storage_balance_rule(m, g, t):
        if g not in m.G_s:
            return Constraint.Skip
        prev = m.soc_init[g] if t==m.T.first() else m.Volume[g,m.T.prev(t)]
        discharge = sum(
            m.Generation[g,e,t] * m.out_frac[g,e]
            for (gg,e) in m.f_out
            if gg==g)
        return m.Volume[g,t] == prev + m.Fuelusetotal[g,t] * m.Fe[g] - discharge
    model.ProductionStorage = Constraint(model.G_s, model.T, rule=storage_balance_rule)

    def charging_max(m, g, t):
        if g not in m.G_s:
            return Constraint.Skip
        return m.Fuelusetotal[g,t] <= m.capacity[g] * m.Charge[g,t]
    model.ChargingStorageMax = Constraint(model.G_s, model.T, rule=charging_max)

    def discharging_max(m, g, t):
        if g not in m.G_s:
            return Constraint.Skip
        discharge = sum(
            m.Generation[g,e,t] * m.out_frac[g,e]
            for (gg,e) in m.f_out
            if gg==g)
        return discharge <= m.capacity[g] * (1-m.Charge[g,t])
    model.DishargingStorageMax = Constraint(model.G_s, model.T, rule=discharging_max)

    def charging_min(m, g, t):
        if g not in m.G_s or m.Minimum[g] <=0 :
            return Constraint.Skip
        return m.Fuelusetotal[g,t] <= m.Minimum[g] * m.Charge[g,t]
    model.ChargingStorageMin = Constraint(model.G_s, model.T, rule=charging_min)

    def discharging_min(m, g, t):
        if g not in m.G_s or m.Minimum[g] <=0:
            return Constraint.Skip
        discharge = sum(
            m.Generation[g,e,t] * m.out_frac[g,e]
            for (gg,e) in m.f_out
            if gg==g)
        return discharge <= m.Minimum[g] * (1-m.Charge[g,t])
    model.DishargingStorageMin = Constraint(model.G_s, model.T, rule=discharging_min)

    def volume_upper_rule(m, g, t):
        return m.Volume[g, t] <= m.soc_max[g]
    model.VolumeUpper = Constraint(model.G_s, model.T, rule=volume_upper_rule)

    def volume_final_soc(m, g):
        return m.Volume[g, m.T.last()] == m.soc_init[g]
    model.TerminalSOC = Constraint(model.G_s, rule=volume_final_soc)


    # 4) Energy balance equations
    def balance_rule(m, a, e, t):
        # 1) GAMS $‐guard: buyE(a,e) OR saleE(a,e) OR any tech at area a with (in or out) of e
        has_buy  = (a,e) in m.buyE
        has_sale = (a,e) in m.saleE
        # find all techs located in area a
        techs_in_area = [tech for (area,tech) in m.location if area==a]
        has_tech = any(
            ((tech,e) in m.f_in) or ((tech,e) in m.f_out)
            for tech in techs_in_area
        )
        if not (has_buy or has_sale or has_tech):
            return Constraint.Skip
        # 2) Left‐hand side: Buy + inbound flows + local generation
        buy_term = m.Buy[a,e,t] if has_buy else 0.0
        inflow   = sum(
            m.Flow[area_in, a, e, t]
            for (area_in, area_to, energy) in m.flowset
            if (area_to==a and energy==e)
        )
        generation = sum(
            m.Generation[tech, e, t]
            for tech in techs_in_area
            if (tech,e) in m.f_out
        )
        # 3) Right‐hand side: local fuel use + Sale + outbound flows
        fueluse   = sum(
            m.Fueluse[tech, e, t]
            for tech in techs_in_area
            if (tech,e) in m.f_in
        )
        sale_term = m.Sale[a,e,t] if has_sale else 0.0
        outflow   = sum(
            m.Flow[a, area_out, e, t]
            for (area_from, area_out, energy) in m.flowset
            if (area_from==a and energy==e)
        )
        # 4) Assemble the balance
        return buy_term + inflow + generation == fueluse + sale_term + outflow
    model.Balance = Constraint(model.A, model.F, model.T, rule=balance_rule)

    def demand_time_rule(m, a, e, t):
        # 1) GAMS $-guard: only if there is any demand at all for (a,e)
        total_area_energy = sum(m.demand[a,e,tt] for tt in m.T)
        if total_area_energy <= 0:
            return Constraint.Skip

        # 2) LHS terms, zero when not defined in the corresponding set
        sale_term = m.Sale[a,e,t] if (a,e) in m.saleE else 0.0
        slack_imp = m.SlackDemandImport[a,e,t] if (a,e) in m.buyE  else 0.0
        slack_exp = m.SlackDemandExport[a,e,t] if (a,e) in m.saleE else 0.0

        lhs = sale_term + slack_imp - slack_exp

        # 3) RHS is the exact demand
        rhs = m.demand[a,e,t]

        # 4) GAMS “=G=” → lhs >= rhs
        return lhs >= rhs

    model.DemandTime = Constraint(model.saleE, model.T, rule=demand_time_rule)






    # 1) MaxBuy
    def max_buy_rule(m, e, t):
        # total capacity for e,t across your interconnector‐areas
        total_cap = sum(
            m.InterconnectorCapacity[a,e,t]
            for a in m.LinesInterconnectors
            if (a,e,t) in m.InterconnectorCapacity.index_set()
        )
        if total_cap <= 0:
            # no lines for this energy/time → skip
            return Constraint.Skip
        # sum of all buys for this energy/time
        lhs = sum(
            m.Buy[a,e,t]
            for (a,f) in m.buyE
            if f == e
        )
        # average capacity per area:
        rhs = total_cap / len(m.LinesInterconnectors)
        return lhs <= rhs
    model.MaxBuy = Constraint(model.F, model.T, rule=max_buy_rule)


    # 2) MaxSale (analogous)
    def max_sale_rule(m, e, t):
        total_cap = sum(
            m.InterconnectorCapacity[a,e,t]
            for a in m.LinesInterconnectors
            if (a,e,t) in m.InterconnectorCapacity.index_set()
        )
        if total_cap <= 0:
            return Constraint.Skip
        lhs = sum(
            m.Sale[a,e,t]
            for (a,f) in m.saleE
            if f == e
        )
        rhs = total_cap / len(m.LinesInterconnectors)
        return lhs <= rhs
    model.MaxSale = Constraint(model.F, model.T, rule=max_sale_rule)


    # --- after you’ve declared Fuelmix & Production ---

    # 1) One example per (tech,energy) from Fuelmix
    print("\n--- Fuelmix (one example per tech–energy) ---")
    seen_fe = set()
    for (g, e, t), con in model.Fuelmix.items():
        if (g,e) in seen_fe:
            continue
        seen_fe.add((g,e))
        print(f"{(g,e,t)} : {con.lower} : {con.body} : {con.upper}")
    print(f"... printed {len(seen_fe)} of {len(model.f_in)} total Fuelmix constraints")

    # 2) One example per (tech,energy) from Production
    print("\n--- Production (one example per tech–energy) ---")
    seen_po = set()
    for (g, e, t), con in model.Production.items():
        if (g,e) in seen_po:
            continue
        seen_po.add((g,e))
        print(f"{(g,e,t)} : {con.lower} : {con.body} : {con.upper}")
    print(f"... printed {len(seen_po)} of {len(model.f_out)} total Production constraints")

    # --- Sample print for MaxBuy: one example per energy ---
    print("\n--- MaxBuy (one example per energy) ---")
    seen_e = set()
    for (e, t), con in model.MaxBuy.items():
        if e in seen_e:
            continue
        seen_e.add(e)
        print(f"  ({e}, {t}) : {con.lower} : {con.body} : {con.upper}")
        if len(seen_e) == len(model.F):
            break
    print(f"... printed {len(seen_e)} of {len(model.F)} energies")

    # --- Likewise for MaxSale ---
    print("\n--- MaxSale (one example per energy) ---")
    seen_e = set()
    for (e, t), con in model.MaxSale.items():
        if e in seen_e:
            continue
        seen_e.add(e)
        print(f"  ({e}, {t}) : {con.lower} : {con.body} : {con.upper}")
        if len(seen_e) == len(model.F):
            break
    print(f"... printed {len(seen_e)} of {len(model.F)} energies")


    # ─── Test: force all non‐storage to max output ────────────────────────
    def force_max_gen(m, g, e, t):
        # only for non‐storage export links
        if (g,e) not in m.f_out or g in m.G_s:
            return Constraint.Skip
        # force Generation = capacity * export‐fraction
        return m.Generation[g,e,t] == m.capacity[g] * m.out_frac[g,e]
    model.ForceMaxGen = Constraint(model.f_out, model.T, rule=force_max_gen)

    # ─── Objective: Maximize total storage discharge ─────────────────────────
    def max_discharge_obj(m):
        return sum(
            m.Generation[g, e, t] * m.out_frac[g, e]
            for (g, e) in m.f_out
            if g in m.G_s
            for t in m.T
        )
    model.MaxDischarge = Objective(rule=max_discharge_obj, sense=maximize)

    # ─── Solve with Gurobi ───────────────────────────────────────────────────
    solver = SolverFactory('gurobi')
    results = solver.solve(model, tee=True)
    model.solutions.load_from(results)   # ensure values are loaded

    # ─── Report storage charge/discharge/SOC/FueluseTotal ─────────────────
    print("\n=== Storage detailed report by hour ===")
    for g in model.G_s:
        # find the energy key for hydrogen in this tech's input‐mix
        hydrogen_energy = next(
            e for (gg, e) in model.f_in
            if gg == g and 'Hydrogen' in e
        )

        print(f"\n--- Storage tech: {g} ---")
        print(f"{'Hour':>8} | {'FuelUseTotal':>12} | {'Charge(H2)':>12} | {'Discharge':>10} | {'SOC':>8}")
        print("-"*62)
        for t in model.T:
            fuel_total    = model.Fuelusetotal[g, t].value or 0.0
            # only the hydrogen share of the total goes into storage
            charge_amt    = fuel_total * model.in_frac[g, hydrogen_energy]
            # actual discharge energy
            discharge_amt = sum(
                model.Generation[g, e, t].value * model.out_frac[g, e]
                for (gg, e) in model.f_out
                if gg == g
            ) or 0.0
            soc           = model.Volume[g, t].value or 0.0

            print(f"{t:>8} | {fuel_total:12.3f} | {charge_amt:12.3f} | {discharge_amt:10.3f} | {soc:8.3f}")



    return

if __name__ == "__main__":
    main()