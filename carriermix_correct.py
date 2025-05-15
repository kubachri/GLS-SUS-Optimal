#!/usr/bin/env python3

import os
import pandas as pd
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Constraint, NonNegativeReals,
    Binary, SolverFactory, Objective, maximize
)
from src.data_loader import load_data

def main():
    # 1) Load all data
    global INC_DIR
    INC_DIR = os.path.join(os.path.dirname(__file__), "inc_data")
    data = load_data()

    # 2) Extract sets and parameters
    techs        = data['G']
    times        = data['T']
    sigma_in     = data['sigma_in']
    sigma_out    = data['sigma_out']
    Pmax         = data['Pmax']
    Pmin_all     = data['Pmin']
    G_s          = [g for g in techs if g in data['G_s']]       # storage techs
    soc_init     = data['SOC_init']
    soc_max      = data['SOC_max']
    storage_fuel = data['storage_fuel']
    profile      = data['Profile']  # dict[(tech,time)] -> fraction in [0,1]

    # 3) Build model
    model = ConcreteModel()

    # --- Sets ---
    model.TECH            = Set(initialize=techs)
    model.TIME            = Set(initialize=times)
    model.IN_SET          = Set(dimen=2,
                                 initialize=[(g,f) for (g,f),v in sigma_in.items() if v>0])
    model.IN_SET_OUT      = Set(dimen=2,
                                 initialize=[(g,f) for (g,f),v in sigma_out.items()
                                             if v>0 and g not in G_s])
    model.OUT_SET_STORAGE = Set(dimen=2,
                                 initialize=[(g,f) for (g,f),v in sigma_out.items()
                                             if v>0 and g in G_s])
    model.G_s             = Set(initialize=G_s)

    # --- Parameters ---
    model.carriermix     = Param(model.IN_SET,
                                 initialize=lambda m,g,f: sigma_in[g,f],
                                 within=NonNegativeReals)
    model.carriermix_out = Param(model.IN_SET_OUT,
                                 initialize=lambda m,g,f: sigma_out[g,f],
                                 within=NonNegativeReals)
    model.carriermix_s   = Param(model.OUT_SET_STORAGE,
                                 initialize=lambda m,g,f: sigma_out[g,f],
                                 within=NonNegativeReals)
    model.PMAX           = Param(model.TECH,
                                 initialize=Pmax,
                                 within=NonNegativeReals)
    model.PMIN_s         = Param(model.G_s,
                                 initialize={g: Pmin_all[g] for g in G_s},
                                 within=NonNegativeReals)
    model.SOC_INIT       = Param(model.G_s,
                                 initialize=soc_init,
                                 within=NonNegativeReals)
    model.SOC_MAX        = Param(model.G_s,
                                 initialize=soc_max,
                                 within=NonNegativeReals)
    model.Profile        = Param(model.TECH, model.TIME,
                                 initialize=lambda m,g,t: profile[g,t],
                                 within=NonNegativeReals)

    # --- Variables ---
    model.FuelTotal   = Var(model.TECH,            model.TIME, domain=NonNegativeReals)
    model.FuelUse     = Var(model.IN_SET,          model.TIME, domain=NonNegativeReals)
    model.Generation  = Var(model.IN_SET_OUT,      model.TIME, domain=NonNegativeReals)
    model.GenerationS = Var(model.OUT_SET_STORAGE, model.TIME, domain=NonNegativeReals)
    model.Charge      = Var(model.G_s,             model.TIME, domain=Binary)
    model.Volume      = Var(model.G_s,             model.TIME, domain=NonNegativeReals)

    # --- Constraints ---

    # 1) Fuel‐mix (input split)
    def fuelmix_rule(model, g, f, t):
        return model.FuelUse[g,f,t] == model.carriermix[g,f] * model.FuelTotal[g,t]
    model.Fuelmix = Constraint(model.IN_SET, model.TIME, rule=fuelmix_rule)

    # 2) Production (non‐storage output)
    def prod_rule(model, g, f, t):
        return model.Generation[g,f,t] == model.carriermix_out[g,f] * model.FuelTotal[g,t]
    model.Production = Constraint(model.IN_SET_OUT, model.TIME, rule=prod_rule)

    # 3) SOC balance for storage
    def soc_rule(model, g, t):
        prev = (model.SOC_INIT[g]
                if t == model.TIME.first()
                else model.Volume[g, model.TIME.prev(t)])
        f_in      = storage_fuel[g]
        charge    = model.FuelUse[g, f_in, t]
        discharge = sum(
            model.GenerationS[g,f,t] * model.carriermix_s[g,f]
            for (gg,f) in model.OUT_SET_STORAGE if gg==g
        )
        return model.Volume[g,t] == prev + charge - discharge
    model.ProductionStorage = Constraint(model.G_s, model.TIME, rule=soc_rule)

    # 4) Charging limits
    def charge_max_rule(model, g, t):
        f_in = storage_fuel[g]
        return model.FuelUse[g,f_in,t] <= model.PMAX[g] * model.Charge[g,t]
    model.ChargingStorageMax = Constraint(model.G_s, model.TIME, rule=charge_max_rule)

    def charge_min_rule(model, g, t):
        f_in = storage_fuel[g]
        return model.PMIN_s[g] * model.Charge[g,t] <= model.FuelUse[g,f_in,t]
    model.ChargingStorageMin = Constraint(model.G_s, model.TIME, rule=charge_min_rule)

    # 5) Discharging limits
    def discharge_max_rule(model, g, f, t):
        return model.GenerationS[g,f,t] <= model.PMAX[g] * (1-model.Charge[g,t])
    model.DischargingStorageMax = Constraint(
        model.OUT_SET_STORAGE, model.TIME, rule=discharge_max_rule)

    def discharge_min_rule(model, g, f, t):
        return model.PMIN_s[g] * (1-model.Charge[g,t]) <= model.GenerationS[g,f,t]
    model.DischargingStorageMin = Constraint(
        model.OUT_SET_STORAGE, model.TIME, rule=discharge_min_rule)

    # 6) Cyclical SOC
    def cyc_rule(model, g):
        return model.Volume[g, model.TIME.last()] == model.SOC_INIT[g]
    model.CyclicalSOC = Constraint(model.G_s, rule=cyc_rule)

    # 7) Generation capacity limit with profile (non-storage only)
    non_storage = [g for g in techs if g not in G_s]
    def gen_cap_rule(model, g, t):
        return model.FuelTotal[g,t] <= model.PMAX[g] * model.Profile[g,t]
    model.GenerationCap = Constraint(non_storage, model.TIME, rule=gen_cap_rule)

    # 8) SOC maximum cap
    def soc_max_rule(model, g, t):
        return model.Volume[g,t] <= model.SOC_MAX[g]
    model.SOCMaxLimit = Constraint(model.G_s, model.TIME, rule=soc_max_rule)

    # 9) Generic storage‐charge ≤ upstream production
    def storage_charge_limit(model, g, t):
        f_in = storage_fuel[g]
        total_prod = sum(
            model.Generation[p, f_in, t]
            for (p, carrier) in model.IN_SET_OUT
            if carrier == f_in
        )
        return model.FuelUse[g, f_in, t] <= total_prod
    model.StorageChargeLimit = Constraint(model.G_s, model.TIME, rule=storage_charge_limit)

    # 10) Objective: maximize total generation
    model.Obj = Objective(
        expr= sum(model.Generation[g,f,t]
                  for (g,f) in model.IN_SET_OUT for t in model.TIME)
            + sum(model.GenerationS[g,f,t]
                  for (g,f) in model.OUT_SET_STORAGE for t in model.TIME),
        sense=maximize
    )

    # 11) Solve
    solver = SolverFactory('gurobi')
    solver.solve(model, tee=True)

    # 12) Export results to Excel
    df_ft = pd.DataFrame({g: {t: model.FuelTotal[g,t].value for t in times}
                          for g in techs})
    idx_fu = pd.MultiIndex.from_tuples(model.IN_SET.data(), names=['tech','fuel'])
    df_fu = pd.DataFrame({t: {(g,f): model.FuelUse[g,f,t].value
                              for g,f in model.IN_SET.data()}
                          for t in times},
                         index=idx_fu)
    idx_gn = pd.MultiIndex.from_tuples(model.IN_SET_OUT.data(), names=['tech','fuel'])
    df_gn = pd.DataFrame({t: {(g,f): model.Generation[g,f,t].value
                              for g,f in model.IN_SET_OUT.data()}
                          for t in times},
                         index=idx_gn)
    idx_gs = pd.MultiIndex.from_tuples(model.OUT_SET_STORAGE.data(), names=['tech','fuel'])
    df_gs = pd.DataFrame({t: {(g,f): model.GenerationS[g,f,t].value
                              for g,f in model.OUT_SET_STORAGE.data()}
                          for t in times},
                         index=idx_gs)
    df_vol = pd.DataFrame({g: {t: model.Volume[g,t].value for t in times}
                           for g in G_s})

    summary = []
    for g in G_s:
        f_in = storage_fuel[g]
        for t in times:
            imp = model.FuelUse[g, f_in, t].value
            exp = sum(model.GenerationS[g,f,t].value
                      for (gg,f) in model.OUT_SET_STORAGE if gg==g)
            soc = model.Volume[g,t].value
            summary.append({'tech':g,'time':t,'import':imp,'export':exp,'soc':soc})
    df_summary = pd.DataFrame(summary).pivot(index='time', columns='tech', values=['import','export','soc'])

    with pd.ExcelWriter('results_storage.xlsx', engine='xlsxwriter') as writer:
        df_ft.to_excel(writer, sheet_name='FuelTotal')
        df_fu.to_excel(writer, sheet_name='FuelUse')
        df_gn.to_excel(writer, sheet_name='Generation')
        df_gs.to_excel(writer, sheet_name='GenStorage')
        df_vol.to_excel(writer, sheet_name='SOC')
        df_summary.to_excel(writer, sheet_name='StorageSummary')

    print("\n✅ results_storage.xlsx written.")

if __name__ == "__main__":
    main()
