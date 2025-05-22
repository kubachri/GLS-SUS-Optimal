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
    G           = data['G']
    T           = data['T']
    sigma_in    = data['sigma_in'].copy()
    sigma_out   = data['sigma_out'].copy()
    Pmax        = data.get('pmax', data['Pmax']).copy()
    profile     = data['Profile']
    G_s         = data['G_s']
    G_p         = [g for g in G if g not in G_s]
    location    = data['location']
    F           = data['F']
    flowset     = data['FlowSet']
    A           = data['A']
    Xcap        = data['Xcap']

    sigma_in  = data['sigma_in'].copy()
    sigma_out = data['sigma_out'].copy()

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

    print('IN_FRAC')
    print('----------')
    print(in_frac)
    print('----------')
    print('OUT_FRAC')
    print('----------')
    print(out_frac)
    print('----------')





if __name__ == "__main__":
    main()