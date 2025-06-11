# src/model/sets.py

from pyomo.environ import Set
from collections import defaultdict

def define_sets(model, data, tech_df):
    """
    Define Pyomo Sets on the model using preprocessed data:
      - A: areas
      - G: all technologies
      - F: fuels/energies
      - T: time periods (possibly sliced in test mode)
      - G_s: storage-only technologies
      - G_p: production-only technologies
      - UC: unit-commitment technologies
      - RR: ramp-rate technologies
      - flowset: inter-area flow links
      - TechToEnergy: mapping of tech→energy exports
      - f_in, f_out: tech-energy import/export pairs
      - buyE, saleE: market buy/sell interfaces
      - location: (area, tech) location pairs
      - LinesInterconnectors: areas with any interconnector capacity
    """

    # Core entity sets
    G_p = [g for g in data['G'] if g not in data['G_s']]

    # Demand
    raw_demand = data['Demand']

    # Technology-to-energy export mapping
    pairs_TE = [
        (g, f)
        for (g, f), out in data['sigma_out'].items()
        if out > 0
    ]

    # Fuel import/export pairs
    pairs_out = [(g, f) for (g, f), out in data['sigma_out'].items() if out > 0]
    pairs_in  = [(g, f) for (g, f), inp in data['sigma_in'].items()  if inp > 0]

    # Market interfaces (areas × fuels with positive prices)
    buy_pairs  = sorted({(a, e) for (a, e, t), p in data['price_buy'].items()  if p > 0})
    sale_pairs = sorted({(a, e) for (a, e, t), p in data['price_sell'].items() if p > 0})

    # Areas that have interconnector capacity
    area_has = defaultdict(bool)
    for (a, f, t), cap in data['Xcap'].items():
        if cap > 0:
            area_has[a] = True
    lines = [a for a, has in area_has.items() if has]

    # Sets definition
    model.A = Set(initialize=data['A'])
    model.G = Set(initialize=data['G'])
    model.F = Set(initialize=data['F'])
    model.T = Set(initialize=data['T'], ordered=True)

    # Storage vs production technologies
    model.G_s = Set(initialize=data['G_s'], within=model.G)
    model.G_p = Set(initialize=G_p, within=model.G)

    # Technology location (area, tech)
    model.location = Set(
        initialize=data['location'],
        dimen=2,
        within=model.A * model.G
    )

    # Unit-commitment and ramp-rate
    model.UC = Set(initialize=data['UC'], within=model.G)
    model.RR = Set(initialize=data['RR'], within=model.G)
    # Inter-area flow definitions
    model.flowset = Set(
        initialize=data['FlowSet'],
        dimen=3,
        within=model.A * model.A * model.F
    )
    model.TechToEnergy = Set(
        initialize=pairs_TE,
        dimen=2,
        within=model.G * model.F
    )
    model.f_out = Set(initialize=pairs_out, dimen=2, within=model.G * model.F)
    model.f_in  = Set(initialize=pairs_in,  dimen=2, within=model.G * model.F)
    model.buyE  = Set(initialize=buy_pairs,  dimen=2, within=model.A * model.F)
    model.saleE = Set(initialize=sale_pairs, dimen=2, within=model.A * model.F)
    model.LinesInterconnectors = Set(initialize=lines, within=model.A)
    model.DemandSet = Set(initialize=raw_demand.keys(), dimen=3, within=model.A * model.F * model.T)

    if model.Demand_Target:
        # Demand Target
        # 1) Compute how many full “168‐step” weeks fit into |T|.
        n_periods = len(data['T'])
        steps_per_week = 168
        n_weeks = n_periods // steps_per_week    # integer division → should be 52

        # 2) Build a Python list [1,2,…,n_weeks]
        week_list = list(range(1, n_weeks + 1))

        # 3) Expose that as a Pyomo Set W:
        model.W = Set(initialize=week_list, ordered=True)