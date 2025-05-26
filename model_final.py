#!/usr/bin/env python3
import os
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Constraint, Binary,
    NonNegativeReals, Objective, minimize, Reals
)
from collections import defaultdict

from src.data_loader import load_data, load_techdata

def main():
    # 1) Load data
    data    = load_data()
    tech_df = load_techdata()

    # 2) Extract sets & raw parameters
    G           = data['G']
    T_all       = data['T']
    sigma_in    = data['sigma_in'].copy()
    sigma_out   = data['sigma_out'].copy()
    G_s         = data['G_s']
    G_p         = [g for g in G if g not in G_s]
    location    = data['location']
    F           = data['F']
    flowset     = data['FlowSet']
    A           = data['A']
    cvar        = data['Cvar']
    cstart      = data['Cstart']

    tech_df.loc[G_s, 'Capacity'] = tech_df.loc[G_s, 'StorageCap']
    capacity = data.get('capacity', data['capacity']).copy()

    # ── 🛠  TEST MODE: limit the horizon ───────────────────────────
    N_test = 5
    def hour_index(h):
        return int(h.split('-')[1])
    T = sorted(T_all, key=hour_index)[:N_test]
    print("Using only these hours:", T)
    # ───────────────────────────────────────────────────────────────

    profile = {(g, t): v for (g, t), v in data['Profile'].items() if t in T}
    demand = {(a, e, t): v for (a, e, t), v in data['Demand'].items() if t in T}
    price_buy = {(a, e, t): v for (a, e, t), v in data['price_buy'].items() if t in T}
    price_sell = {(a, e, t): v for (a, e, t), v in data['price_sell'].items() if t in T}
    Xcap = {(a, f, t): v for (a, f, t), v in data['Xcap'].items() if t in T}


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

    model.capacity.pprint()

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

    # 5) Demand constraint
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

    # 6) MaxBuy
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

    # 7) MaxSale (analogous)
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

    # 8) Ramp-up: (Generation[t] – Generation[t-1])/Fe ≤ LHS
    def ramp_up_rule(m, g, t):
        # only where a nonzero RampRate exists
        if m.RampRate[g] <= 0:
            return Constraint.Skip
        # skip the first hour (no t-1)
        if t == m.T.first():
            return Constraint.Skip
        # build left‐hand side exactly as GAMS:
        # if not UC: just rampRate
        # if UC: rampRate*Online[t-1] + Minimum*(1-Online[t-1])
        prev_on = m.Online[g, m.T.prev(t)]
        if g not in m.UC:
            lhs = m.RampRate[g]
        else:
            lhs = m.RampRate[g]*prev_on + m.Minimum[g]*(1-prev_on)
        # right‐hand side: sum over export‐energies of (Gen[t]–Gen[t-1])/Fe
        rhs = sum(
            (m.Generation[g,e,t] - m.Generation[g,e,m.T.prev(t)])/m.Fe[g]
            for (gg,e) in m.f_out
            if gg == g
        )
        return lhs >= rhs
    model.RampUp = Constraint(model.G, model.T, rule=ramp_up_rule)

    # 9) Ramp-down: (Generation[t-1] – Generation[t])/Fe ≤ LHS
    def ramp_down_rule(m, g, t):
        if m.RampRate[g] <= 0:
            return Constraint.Skip
        if t == m.T.first():
            return Constraint.Skip
        # LHS: if not UC, rampRate; else rampRate*Online[t] + Minimum*(1-Online[t])
        if g not in m.UC:
            lhs = m.RampRate[g]
        else:
            lhs = m.RampRate[g]*m.Online[g,t] + m.Minimum[g]*(1-m.Online[g,t])
        # RHS: sum over export‐energies of (Gen[t-1]–Gen[t])/Fe
        rhs = sum(
            (m.Generation[g,e,m.T.prev(t)] - m.Generation[g,e,t]) / m.Fe[g]
            for (gg,e) in m.f_out
            if gg == g
        )
        return lhs >= rhs
    model.RampDown = Constraint(model.G, model.T, rule=ramp_down_rule)

    # 10) Capacity constraint: Capacity*Online ≥ FuelUseTotal  (only for UC)
    def capacity_rule(m, g, t):
        if g not in m.UC:
            return Constraint.Skip
        return m.capacity[g] * m.Online[g,t] >= m.Fuelusetotal[g,t]
    model.Capacity = Constraint(model.G, model.T, rule=capacity_rule)

    # 11) Minimum-load: FuelUseTotal ≥ Minimum*Online  (only if Minimum>0)
    def minimum_load_rule(m, g, t):
        if g not in m.UC or m.Minimum[g] <= 0:
            return Constraint.Skip
        return m.Fuelusetotal[g,t] >= m.Minimum[g] * m.Online[g,t]
    model.MinimumLoad = Constraint(model.G, model.T, rule=minimum_load_rule)

    # 12) Startup cost: Startcost ≥ StartupCost*(Online[t]–Online[t-1])  (only if cstart>0)
    def startup_cost_rule(m, g, t):
        if g not in m.UC or m.cstart[g] <= 0:
            return Constraint.Skip
        # treat previous‐hour Offline before t=first as 0
        prev_on = 0 if t == m.T.first() else m.Online[g, m.T.prev(t)]
        return m.Startcost[g,t] >= m.cstart[g] * (m.Online[g,t] - prev_on)
    model.StartupCost = Constraint(model.G, model.T, rule=startup_cost_rule)

    # 13) Objective Function
    # 1) Penalty for unmet demand or over-supply
    penalty = 1e6   # for example

    # 2) Cost‐definition constraint: mirror your GAMS “Cost =E= …”
    def cost_definition_rule(m):
        # a) Fuel cost (imports are a positive cost → negative in objective)
        imp_cost = sum(
            m.price_buy[a,e,t] * m.Buy[a,e,t]
            for (a,e) in m.buyE
            for t in m.T
        )
        # b) Sale revenue
        sale_rev = sum(
            m.price_sale[a,e,t] * m.Sale[a,e,t]
            for (a,e) in m.saleE
            for t in m.T
        )
        # c) Variable O&M on all tech→energy links
        var_om = sum(
            m.Generation[g,e,t] * m.cvar[g]
            for (g,e) in m.TechToEnergy
            for t in m.T
        )
        # d) Startup costs
        startup = sum(
            m.Startcost[g,t]
            for g in m.G
            for t in m.T
        )
        # e) Slack penalties (both import‐slack and export‐slack)
        slack = (
            sum(m.SlackDemandImport[a,e,t] for (a,e) in m.buyE  for t in m.T)
          + sum(m.SlackDemandExport[a,e,t] for (a,e) in m.saleE for t in m.T)
        )

        # GAMS: Cost =E= -imp_cost + sale_rev - var_om - startup - penalty*slack
        return m.Cost == (
           - imp_cost
           + sale_rev
           - var_om
           - startup
           - penalty * slack
        )
    model.CostDefinition = Constraint(rule=cost_definition_rule)
    # 3) Objective: minimize the Cost variable
    model.Obj = Objective(expr=model.Cost, sense=minimize)

    return model


if __name__ == "__main__":
    from pyomo.environ import SolverFactory, value
    from src.excel_writter import export_results_to_excel

    m = main()   # make sure main() returns the model!

    print("\n--- Sample constraint rows (one per “prefix”) ---\n")
    for c in m.component_objects(Constraint, active=True):
        comp = getattr(m, c.name)
        print(f"Constraint: {c.name}")
        if comp.is_indexed():
            seen = set()
            for idx in comp:
                prefix = idx[:-1] if isinstance(idx, tuple) else idx
                if prefix in seen:
                    continue
                seen.add(prefix)
                row = comp[idx]
                print(f"  {c.name}{idx} : {row.body}")
        else:
            # scalar constraint → just print its bounds, not the huge body
            lb = comp.lower() if comp.has_lb() else None
            ub = comp.upper() if comp.has_ub() else None
            print(f"  {c.name} : ", end="")
            if lb is not None:  print(f"LB={lb}  ", end="")
            if ub is not None:  print(f"UB={ub}", end="")
            print()
        print()

    # --- Now solve ---
    solver = SolverFactory('gurobi')   # or 'glpk'
    results = solver.solve(m, tee=True)
    m.solutions.load_from(results)

    print(f"\n>> Objective = {value(m.Obj):.3f}")

    print("\n--- GAMS-style Sample equations (one per prefix) ---\n")
    for cname, con in m.component_map(Constraint).items():
        if not con.active:
            continue
        # pick an example index (the first) for this constraint
        if con.is_indexed():
            idx = next(iter(con))
            cdata = con[idx]
            # only print each constraint name once
            header = f"{cname}({','.join(str(i) for i in idx)})"
        else:
            idx = None
            cdata = con
            header = cname

        # figure out sense and RHS
        lb = cdata.lower
        ub = cdata.upper
        if lb is not None and ub is not None and lb == ub:
            sense = "=E="
            rhs = lb
        elif ub is not None:
            sense = "=L="
            rhs = ub
        elif lb is not None:
            sense = "=G="
            rhs = lb
        else:
            sense = "??"
            rhs = 0

        body_str = str(cdata.body)  # e.g. "- Fueluse[...] + Generation[...] - Flow[...]"
        val = value(cdata.body)

        print(f"{header}..  {body_str}  {sense}  {rhs} ;   (LHS = {val:.6g})")

    export_results_to_excel()