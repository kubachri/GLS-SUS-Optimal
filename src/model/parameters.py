# src/model/params.py

from pyomo.environ import Param, NonNegativeReals, Reals, PositiveIntegers

def define_params(model, data, tech_df):
    """
    Attach all Model Param objects:
     - capacity, Profile, demand, price_buy, price_sale, InterconnectorCapacity
     - cstart, cvar, RampRate, Minimum, soc_init, soc_max
     - Fe, in_frac, out_frac
    """
    # Unpack raw data
    sigma_in    = data['sigma_in'].copy()
    sigma_out   = data['sigma_out'].copy()
    G_s         = data['G_s']
    capacity = data['capacity']
    original_capacity = data['original_cap']

    # 2) Efficiency Fe per tech
    Fe = {}
    for g in tech_df.index:
        tot_in  = sum(v for (gg,e), v in sigma_in.items()  if gg==g)
        tot_out = sum(v for (gg,e), v in sigma_out.items() if gg==g)
        Fe[g] = (tot_out/tot_in) if tot_in>0 else 1.0

    # 3) Mix fractions
    in_frac  = {}
    out_frac = {}
    for (g,e), v in sigma_in.items():
        if v>0:
            total = sum(val for (gg,ee), val in sigma_in.items() if gg==g)
            in_frac[(g,e)] = v/total
    for (g,e), v in sigma_out.items():
        if v>0:
            total = sum(val for (gg,ee), val in sigma_out.items() if gg==g)
            out_frac[(g,e)] = v/total

    # 4) Storage parameters
    soc_init = {g: tech_df.at[g, 'InitialVolume'] for g in G_s}
    soc_max  = {g: tech_df.at[g, 'StorageCap']     for g in G_s}

    # 5) Cost & startup data
    cvar   = tech_df['VariableOmcost'].astype(float).to_dict()
    cstart = tech_df['StartupCost'].astype(float).to_dict()

    # 6) Ramping & minimum
    RampRate = tech_df['RampRate'].astype(float).to_dict()
    Minimum  = tech_df['Minimum'].astype(float).to_dict()

    # 7) Time‐series & interconnector capacities
    profile     = data['Profile']
    demand      = data['Demand']
    price_buy   = data['price_buy']
    price_sell  = data['price_sell']
    Xcap        = data['Xcap']

    # === Now attach all to the model ===
    model.Profile = Param(model.G, model.T, initialize=profile, within=NonNegativeReals)
    model.capacity = Param(model.G, initialize=capacity, within=NonNegativeReals)
    model.original_capacity = Param(model.G, initialize=original_capacity, within=NonNegativeReals)
    model.Fe       = Param(model.G,          initialize=Fe,         within=NonNegativeReals)
    model.soc_init = Param(model.G_s,        initialize=soc_init,   within=NonNegativeReals)
    model.soc_max  = Param(model.G_s,        initialize=soc_max,    within=NonNegativeReals)
    model.cstart   = Param(model.G, initialize=cstart, within=NonNegativeReals)
    model.cvar = Param(model.G, initialize=cvar, within=NonNegativeReals)
    model.RampRate = Param(model.G, initialize=RampRate, within=NonNegativeReals)
    model.Minimum = Param(model.G, initialize=Minimum, within=NonNegativeReals)
    model.in_frac  = Param(model.G, model.F, initialize=in_frac, within=NonNegativeReals)
    model.out_frac = Param(model.G, model.F, initialize=out_frac, within=NonNegativeReals)
    model.demand = Param(model.DemandSet, initialize=demand, within=NonNegativeReals)
    model.price_buy = Param(model.A, model.F, model.T, initialize=price_buy, within=Reals)
    model.price_sale = Param(model.A, model.F, model.T, initialize=price_sell, within=Reals)
    model.InterconnectorCapacity = Param(model.LinesInterconnectors, model.F, model.T,
                                         initialize=Xcap, default= 0, within=NonNegativeReals)

    if model.Demand_Target:
        # 1) Define weekOfT[t] = integer in {1, 2, …, n_weeks}
        #    where each week covers 168 time‐steps in T.
        def _week_of_t_init(m, t):
            # build an ordered Python list of all time‐indices
            all_t = list(m.T)            # e.g. [t1, t2, …, tN]
            idx = all_t.index(t)         # gives 0, 1, 2, …, N-1
            return (idx // 168) + 1      # integer division → 1..n_weeks

        # Attach the Param to the model
        model.weekOfT = Param(model.T, initialize=_week_of_t_init, within=PositiveIntegers)

        # 2) Define methanol_demand_week[w] = 32000/52  for each week w in W
        annual_methanol_demand = 8000.0
        new_target = annual_methanol_demand / 8760.0 * len(model.T)
        weekly_target = float(new_target / len(model.W))
        demand_dict = { w: weekly_target for w in model.W}
        model.methanol_demand_week = Param(model.W, initialize=demand_dict, within=NonNegativeReals)
        print(f"Weeekly demand target: {int(weekly_target)} tons per week.\n")

        #Biomethane demand
        per_hour_demand = 19000000 / 8760
        model.biomethane_demand_hour = Param(model.T, initialize={t: per_hour_demand for t in model.T})
