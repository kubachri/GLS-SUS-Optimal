# scripts/run_model.py
import argparse
from pyomo.environ import SolverFactory, Suffix, value, Var, Binary
from src.config           import ModelConfig
from src.model.builder    import build_model
from src.utils.export_resultT import export_results
from src.model.objective import debug_objective
from pyomo.repn import generate_standard_repn
from pyomo.core.base.constraint import Constraint
import csv
import time
from datetime import datetime, timedelta
from src.utils.max_contraint_violation import detect_max_constraint_violation
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test', action='store_true', help="short horizon test run")
    p.add_argument('--n-test', type=int, help="hours to keep when --test is on")
    p.add_argument('--penalty', type=float, help="penalty multiplier for slack in objective")
    p.add_argument('--data', type=str, help = "name of the folder under project root to use for 'inc_data_*'")
    return p.parse_args()

def main():

    start_time = time.time()
    print("==========================")
    print("Model Run Started")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("==========================\n")

    args = parse_args()
    defaults = ModelConfig()
    cfg = ModelConfig(
        test_mode=args.test,
        n_test=args.n_test if args.n_test is not None else defaults.n_test,
        penalty=args.penalty if args.penalty is not None else defaults.penalty,
        data=args.data if args.data is not None else defaults.data
    )

    print("Building Pyomo model ...\n")
    model = build_model(cfg)
    print("Model built successfully.\n")

    if model.Demand_Target:
        # 2) Tell Pyomo we want to import duals (for later LP)
        model.dual = Suffix(direction=Suffix.IMPORT)

    # 3) Solve the MIP
    solver = SolverFactory('gurobi_persistent')
    solver.set_instance(model)
    solver.options['MIPGap'] = 0.0015
    print("Solving MIP …\n")


    mip_start = time.time()
    mip_result = solver.solve(model, tee=True)
    mip_end = time.time()
    print("MIP solve finished.\n")
    mip_elapsed = mip_end - mip_start
    if mip_elapsed < 60:
        print(f"MIP Solve time: {int(mip_elapsed)} seconds\n")
    else:
        mins, secs = divmod(int(mip_elapsed), 60)
        print(f"MIP Solve time: {mins:02d}:{secs:02d} (mm:ss)\n")

    termination = str(mip_result.solver.termination_condition).lower()
    if "infeasible" in termination:
        print("\nModel reported infeasible. Attempting IIS extraction...\n")
        from src.utils.infeasibilities import compute_gurobi_iis
        compute_gurobi_iis(model, solver)
        return  # Stop further pipeline execution if infeasible

    print("Checking constraint violations after MIP solve...")
    detect_max_constraint_violation(model, threshold=1e-4, top_n=10)

    # Collect debug info
    rows = []
    for g in model.G:
        name = g.lower()
        if 'wind' in name or 'solar' in name:
            # find the electricity export (g,'Electricity') parameters
            # you could generalize if your out_frac key is different
            if ('Electricity' not in [e for (gg, e) in model.f_out if gg == g]):
                continue
            out_frac = model.out_frac[g, 'Electricity']
            print('out frac', out_frac)
            eff = model.Fe[g]

            for idx, t in enumerate(model.T):
                if idx >= 25:  # only hours 1–100
                    break
                cap = model.capacity[g]
                prof = model.Profile[g, t]
                max_fuel = cap * prof  # max pre‐efficiency fuel
                max_el = max_fuel * eff * out_frac  # what that fuel could produce
                fuel_act = model.Fuelusetotal[g, t].value  # actual fuel used
                gen_act = model.Generation[g, 'Electricity', t].value
                efficiency = model.Fe[g
                ]
                rows.append({
                    'tech': g,
                    'time': t,
                    'cap*profile': max_fuel,
                    'fuel_used': fuel_act,
                    'max_electricity': max_el,
                    'actual_electricity': gen_act,
                    'efficiency': efficiency
                })

    # Build DataFrame and display or save
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    # Or: df.to_csv('wind_solar_debug.csv', index=False)
    print()
    if model.Demand_Target:

        # After solving the MIP, but before fixing binaries:
        print("Fixing binaries before LP dual extraction ...\n")

        seen = set()
        for varobj in model.component_data_objects(Var, descend_into=True):
            if varobj.domain is Binary and varobj.value is not None:
                comp_name = varobj.parent_component().name  # e.g. "Charge" or "Online"
                tech = varobj.index()[0]  # first index = technology name
                seen.add((comp_name, tech))

        # Now print one line per (component, tech)
        for comp_name, tech in sorted(seen):
            print(f"{comp_name}  →  {tech}")

        # Finally, fix all binaries as before
        for varobj in model.component_data_objects(Var, descend_into=True):
            if varobj.domain is Binary and varobj.value is not None:
                varobj.fix(varobj.value)

        # 5) Clear any old duals, then re‐solve as an LP to get duals
        print("Re‐solving as an LP to extract duals …\n")
        lp_start = time.time()
        model.dual.clear()
        lp_result = solver.solve(model, tee=False)
        lp_end = time.time()
        print("LP solve finished.")
        lp_elapsed = lp_end - lp_start
        if lp_elapsed < 60:
            print(f"LP Solve time: {int(lp_elapsed)} seconds\n")
        else:
            mins, secs = divmod(int(lp_elapsed), 60)
            print(f"LP Solve time: {mins:02d}:{secs:02d} (mm:ss)\n")
        print(f"LP termination condition: {lp_result.solver.termination_condition}\n")

    # export_results_to_excel(model)
    print("Exporting results to Excel ...")
    export_results(model, cfg)
    print("Results exported successfully.")
    debug_objective(model, cfg)

    elapsed = time.time() - start_time
    print("\n==========================")
    print("Pyomo Model Run Completed")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    elapsed_td = timedelta(seconds=int(elapsed))
    print(f"Total runtime: {elapsed_td}")

if __name__ == '__main__':
    main()
