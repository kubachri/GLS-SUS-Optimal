#!/usr/bin/env python3
import os
import pandas as pd
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

     # 3) Build model
    model = ConcreteModel()

    # --- Sets ---
    model.A = Set(initialize=A)
    model.G = Set(initialize=G)
    model.F = Set(initialize=F)
    model.T = Set(initialize=T)
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
    model.InterconnectorCapacity = Param(model.LinesInterconnectors, model.F, model.T, initialize=Xcap, within=NonNegativeReals)

   # --- Variables ---


if __name__ == "__main__":
    main()