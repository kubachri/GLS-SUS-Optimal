import pandas as pd
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Constraint, Binary, Reals,
    NonNegativeReals, SolverFactory, Objective, maximize, value
)
from src.data_loader import load_data, load_techdata
from src.excel_writter import export_results_to_excel

def main():
    # 1) Load data
    data    = load_data()
    tech_df = load_techdata()

    # 2) Extract sets & raw parameters
    G           = data['G']
    T           = data['T']
    sigma_in    = data['sigma_in'].copy()
    sigma_out   = data['sigma_out'].copy()
    Pmax        = data.get('pmax', data['Pmax']).copy()
    profile     = data['Profile']
    # 2.a) All energy carriers, sorted
    carriers    = sorted({f for (g, f) in sigma_in} | {f for (g, f) in sigma_out})
    G_s         = data['G_s']
    A    = data['A']            # list of areas
    flow = data['FlowSet']
    price_buy  = data['price_buy']   # dict (area, energy, time) → import price
    price_sell = data['price_sell']  # dict (area, energy, time) → export price
    demand     = data['Demand']      # dict (area, energy, time) → demand
    Xcap       = data['Xcap']        # dict (area,energy,time) → interconnector capacity

    # ─── 2.1) UC / RR / scale capacity ─────────────────────────────────
    orig_cap = tech_df['Capacity'].copy()
    orig_min = tech_df['Minimum'].copy()
    orig_ramp = tech_df['RampRate'].copy()
    sum_in_raw = {
        g: sum(v for (gg, e), v in sigma_in.items() if gg == g)
        for g in tech_df.index
    }
    sum_out_raw = {
        g: sum(v for (gg, e), v in sigma_out.items() if gg == g)
        for g in tech_df.index
    }

    UC = [g for g in tech_df.index if orig_min[g] > 0]
    RR = [g for g in tech_df.index if orig_ramp[g] > 0]

    for g in UC:
        tech_df.at[g, 'Minimum'] = sum_in_raw[g] * orig_cap[g] * orig_min[g]
    for g in RR:
        tech_df.at[g, 'RampRate'] = sum_in_raw[g] * orig_cap[g] * orig_ramp[g]
    for g in tech_df.index:
        newc = sum_in_raw[g] * orig_cap[g]
        tech_df.at[g, 'Capacity'] = newc
        Pmax[g] = newc

    # ─────────────────────────────────────────────────────────────────
    # 2.b) Fe for every technology
    Fe = {}
    for g in tech_df.index:
        tot_in = sum(v for (gg, e), v in sigma_in.items() if gg == g)
        tot_out = sum(v for (gg, e), v in sigma_out.items() if gg == g)
        Fe[g] = (tot_out / tot_in)

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

    # 2.e) SOC init & max
    soc_init = {g: tech_df.at[g, 'InitialVolume'] for g in G_s}
    soc_max = {g: tech_df.at[g, 'StorageCap'] for g in G_s}

    #Non-storage technologies
    nonstor = [(g,f) for (g,f) in out_frac.keys() if g not in G_s]

    print(flow)

    # 3) Build model
    model = ConcreteModel()

    # --- Sets ---
    model.G = Set(initialize=G)
    model.F = Set(initialize=carriers)
    model.T = Set(initialize=T)
    model.G_s = Set(initialize=G_s, within=model.G)

    model.IN = Set(initialize=in_frac.keys(),
                     dimen=2,
                     within=model.G * model.F)
    model.OUT = Set(initialize=out_frac.keys(),
                     dimen=2,
                     within=model.G * model.F)

    #for non-storage techs
    model.OUT_nonstor = Set(initialize=nonstor, dimen=2, within=model.G*model.F)

    model.A = Set(initialize=data['A'])
    model.Flow = Set(initialize=flow, dimen=3)

    def _can_buy(m,a,e):
        return any(m.price_buy[a,e,t]>0 for t in m.T)
    def _can_sell(m,a,e):
        return any(m.price_sell[a,e,t]>0 for t in m.T)

    model.buyE  = Set(initialize=[(a,e) for a in model.A for e in model.F if _can_buy(model,a,e)], dimen=2)
    model.saleE = Set(initialize=[(a,e) for a in model.A for e in model.F if _can_sell(model,a,e)], dimen=2)


    # --- Parameters ---
    model.PMAX = Param(model.G, initialize=Pmax, within=NonNegativeReals)
    model.Profile = Param(model.G, model.T, initialize=profile, within=NonNegativeReals)
    model.Fe = Param(model.G, initialize=Fe, within=NonNegativeReals)
    model.soc_init = Param(model.G_s, initialize=soc_init, within=NonNegativeReals)
    model.soc_max = Param(model.G_s, initialize=soc_max, within=NonNegativeReals)

    # each element = (area_from, area_to, energy)
    model.price_buy  = Param(model.A, model.F, model.T, initialize=price_buy,  default=0.0)
    model.price_sell = Param(model.A, model.F, model.T, initialize=price_sell, default=0.0)
    model.demand     = Param(model.A, model.F, model.T, initialize=demand,     default=0.0)
    model.Xcap       = Param(model.A, model.F, model.T, initialize=Xcap,       default=0.0)


    model.in_frac = Param(model.IN,
                          initialize=in_frac,
                          within=NonNegativeReals)
    model.out_frac = Param(model.OUT,
                           initialize=lambda m, g, f: out_frac.get((g, f)),
                           within=NonNegativeReals)


    # --- Variables ---
    model.FuelUseTotal     = Var(model.G,   model.T,      within=NonNegativeReals)
    model.FuelUse = Var(model.IN, model.T, within=NonNegativeReals)
    model.Generation = Var(model.OUT, model.T, within=NonNegativeReals)
    model.Volume = Var(model.G_s, model.T, bounds= lambda m,g,t: (0, m.soc_max[g]), within=NonNegativeReals)
    model.Charge = Var(model.G_s, model.T, domain=Binary)
    # how much we import into each *area* for each energy & time
    model.Buy  = Var(model.buyE,  model.T, within=NonNegativeReals)
    model.Sale = Var(model.saleE, model.T, within=NonNegativeReals)

    # inter‐area flows
    model.FlowAmt = Var(model.Flow, model.T, within=NonNegativeReals)

    # can be negative, or split into two nonnegatives if you prefer
    # storage
    model.FuelTotalSto  = Var(model.G_s,   model.T,      within=NonNegativeReals)
    model.FuelUseSto    = Var(model.STO_IN, model.T,      within=NonNegativeReals)
    model.Discharge     = Var(model.G_s,   model.T,      within=NonNegativeReals)
    model.GenerationSto = Var(model.STO_OUT,model.T,      within=NonNegativeReals)
    model.Volume        = Var(model.G_s,   model.T,      within=NonNegativeReals)
    model.Mode          = Var(model.G_s,   model.T,      domain=Binary)

    # --- Constraints ---

    # 4.1) prod fuel‐mix
    def prod_fuelmix(m, g, f, t):
        return m.FuelUse[g, f, t] == m.in_frac[g, f] * m.FuelUseTotal[g, t]

    model.ProdFuelMix = Constraint(model.IN, model.T, rule=prod_fuelmix)

    # 4.2) prod output only for non-storage techs
    def prod_out(m, g, f, t):
        return m.Generation[g, f, t] \
            == m.out_frac[g, f] * m.FuelUseTotal[g, t] * m.Fe[g]
    model.ProdOut = Constraint(model.OUT, model.T, rule=prod_out)

    # 4.3) cap production
    def cap_prod(m, g, t):
        return model.FuelUseTotal[g, t] <= model.PMAX[g] * model.Profile[g, t]

    model.CapProd = Constraint(model.G, model.T, rule=cap_prod)

    def soc_level(m, g, t):
        # 1) previous SoC
        if t == m.T.first():
            prev = m.soc_init[g]
        else:
            prev = m.Volume[g, m.T.prev(t)]
        # 2) inflow
        inflow = m.FuelUseTotal[g, t] * m.Fe[g]
        # 3) outflow: only those (g,e) in your out_frac/export set
        outflow = sum(
            m.Generation[g, e, t] * m.out_frac[g, e]
            for (gg, e) in m.OUT
            if gg == g
        )
        return m.Volume[g, t] == prev + inflow - outflow

    model.SOCBalance = Constraint(model.G_s, model.T, rule=soc_level)

    def charging_max(m, g, t):
        return m.FuelUseTotal[g, t] <= m.PMAX[g] * m.Charge[g, t]
    model.ChargingStorageMax = Constraint(model.G_s, model.T, rule=charging_max)

    def discharging_max(m, g, t):
        outflow = sum(
            m.Generation[g, e, t] * m.out_frac[g, e]
            for (gg, e) in m.OUT
            if gg == g
        )
        return outflow <= m.PMAX[g] * (1-m.Charge[g, t])
    model.DischargingStorageMax = Constraint(model.G_s, model.T, rule=discharging_max)

    def storage_cycle(m, g):
        t_last = m.T.last()
        return m.Volume[g, t_last] == m.soc_init[g]

    model.StorageCycle = Constraint(model.G_s, rule=storage_cycle)

    #energy balance
    def balance_rule(m, a, e, t):
        # imports into area a
        b = m.Buy[a,e,t] if (a,e) in m.buyE else 0

        # inbound inter‐area
        inflows = sum(m.FlowAmt[a0,a,e,t]
                      for (a0,a1,e2) in m.Flow if a1==a and e2==e)

        # tech‐out in area
        tech_out = sum(m.Generation[g,f,t]
                       for (g,f) in m.OUT_nonstor
                       if f==e and data['area_of'][g]==a)

        # =E=
        # tech‐in in area
        tech_in  = sum(m.FuelUse[g,f,t]
                       for (g,f) in m.IN
                       if f==e and data['area_of'][g]==a)

        # sales from area
        s = m.Sale[a,e,t] if (a,e) in m.saleE else 0

        # outbound inter‐area
        outflows = sum(m.FlowAmt[a,a1,e,t]
                       for (a,a1,e2) in m.Flow if e2==e)

        return b + inflows + tech_out == tech_in + s + outflows

    model.Balance = Constraint(model.A, model.F, model.T, rule=balance_rule)


    def demand_rule(m,a,e,t):
        # sale + slack - sale_down etc.
        return m.Sale[a,e,t] >= m.demand[a,e,t]
    model.DemandCon = Constraint(model.saleE, model.T, rule=demand_rule)

    def flow_capacity(m, a1, a2, e, t):
        return m.FlowAmt[a1,a2,e,t] <= m.Xcap[a1,e,t]
    model.FlowCap = Constraint(model.Flow, model.T, rule=flow_capacity)

    def capacity_link(m, g, t):
        return m.FuelUseTotal[g,t] == m.PMAX[g] * m.Online[g,t]
    model.Capacity = Constraint(UC, model.T, rule=capacity_link)

    def min_load(m, g, t):
        return m.PMIN[g] * m.Online[g,t] <= m.FuelUseTotal[g,t]
    model.MinLoad = Constraint(UC, model.T, rule=min_load)

    def ramp_up(m, g, t):
        if t==m.T.first(): return Constraint.Skip
        return (sum(m.Generation[g,e,t] for (gg,e) in m.OUT if gg==g) / m.Fe[g]
                - sum(m.Generation[g,e,m.T.prev(t)] for (gg,e) in m.OUT if gg==g) / m.Fe[g]) \
               <= m.Rup[g]
    model.RampUp = Constraint(RR, model.T, rule=ramp_up)

    def ramp_down(m, g, t):
        if t==m.T.first(): return Constraint.Skip
        return (sum(m.Generation[g,e,m.T.prev(t)] for (gg,e) in m.OUT if gg==g) / m.Fe[g]
                - sum(m.Generation[g,e,t] for (gg,e) in m.OUT if gg==g) / m.Fe[g]) \
               <= m.Rdown[g]
    model.RampDown = Constraint(RR, model.T, rule=ramp_down)

    def startup_cost(m, g, t):
        if t==m.T.first(): return Constraint.Skip
        return m.StartCost[g,t] == data['Cstart'][g] * (m.Online[g,t] - m.Online[g,m.T.prev(t)])
    model.StartupCost = Constraint(UC, model.T, rule=startup_cost)


    model.obj = Objective(
        expr = (
           - sum(m.price_buy[a,e,t]*m.Buy[a,e,t]   for (a,e) in m.buyE for t in m.T)
           + sum(m.price_sell[a,e,t]*m.Sale[a,e,t] for (a,e) in m.saleE for t in m.T)
           - sum(m.Generation[g,f,t]*data['Cvar'][g]
                 for (g,f) in m.OUT for t in m.T)
           - sum(m.StartCost[g,t] for g in UC for t in m.T)
           # – penalty*sum(Slack) if you add slack
        ),
        sense = maximize
    )



    # solve …
    solver = SolverFactory('gurobi')
    solver.solve(model, tee=True)

    filepath = export_results_to_excel(model)
    print(f"[Download results]({filepath})")

    hours = [1, 2, 10, 167, 168]
    hours_to_report = [f"Hour-{h}" for h in hours]

    for (g, f) in model.OUT:
        for t in hours_to_report:
            print(f"{g},{f},{t} →", value(model.Generation[g, f, t]))

    # collect total production for selected hours

    total_by_hour = {}
    for t in hours_to_report:
        total_by_hour[t] = sum(
            model.Generation[g, f, t].value
            for (g, f) in model.OUT
        )

    df = pd.DataFrame.from_dict(
        total_by_hour, orient='index', columns=['TotalGeneration']
    )
    df.index.name = 'Hour'
    print(df)

    # selected_tech = 'SolarPV'
    # print("\n=== ProdOut values ===")
    # for (g, f) in model.OUT:
    #         if g==selected_tech:
    #             for t in model.T:
    #                 gen = value(model.Generation[g, f, t])
    #                 tot = value(model.FuelUseTotal[g, t])
    #                 if gen != 0 or tot != 0:
    #                     print(f"ProdOut[{g},{f},{t}]:  Generation = {gen:.6f},  FuelUseTotal = {tot:.6f}")
    #
    # print("\n=== ProdFuelMix values ===")
    # for (g, f) in model.IN:
    #     if g == selected_tech:
    #         for t in model.T:
    #             fuel = value(model.FuelUse[g, f, t])
    #             tot = value(model.FuelUseTotal[g, t])
    #             if fuel != 0 or tot != 0:
    #                 print(f"ProdFuelMix[{g},{f},{t}]:  FuelUse = {fuel:.6f},  FuelUseTotal = {tot:.6f}")

if __name__ == "__main__":
    main()
