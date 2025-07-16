# scripts/run_model.py
import argparse
from pyomo.environ import SolverFactory, Suffix, value, Var, Binary, TransformationFactory, Constraint
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
    p.add_argument('--demand-target', type=lambda x: x.lower() == 'true', help="choose whether to enforce methanol annual demand target")
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
        data=args.data if args.data is not None else defaults.data,
        demand_target=args.demand_target if args.demand_target is not None else defaults.demand_target
    )

    print("Building Pyomo model ...\n")
    model = build_model(cfg)
    print("Model built successfully.\n")


    model.dual = Suffix(direction=Suffix.IMPORT)

    # 3) Solve the MIP
    solver = SolverFactory('gurobi_persistent')
    solver.set_instance(model)
    solver.options['MIPGap'] = 0.05
    print("\nSolving MIP …\n")
    mip_result = solver.solve(model, tee=True)
    mip_obj = value(model.Obj)
    print(f"→ MIP objective (profit) = {mip_obj:,.2f}\n")
    print("\nMIP solve finished.\n")

    termination = str(mip_result.solver.termination_condition).lower()
    if "infeasible" in termination:
        print("\nModel reported infeasible. Attempting IIS extraction...\n")
        from src.utils.infeasibilities import compute_gurobi_iis
        compute_gurobi_iis(model, solver)
        return  # Stop further pipeline execution if infeasible

    print("Checking constraint violations after MIP solve...")
    detect_max_constraint_violation(model, threshold=1e-4, top_n=10)

    # After solving the MIP, but before fixing binaries:
    for v in model.component_data_objects(Var, descend_into=True):
        if v.domain is Binary and v.value is not None:
            v.fix(v.value)

    print("Relaxing integer vars → pure LP …\n")
    TransformationFactory('core.relax_integer_vars').apply_to(model)

    # 5) Clear any old duals, then re‐solve as an LP to get duals
    print("Re‐solving as an LP to extract duals …\n")
    lp_solver = SolverFactory('gurobi')
    lp_result = lp_solver.solve(model, tee=False, suffixes=['dual'])
    lp_obj = value(model.Obj)
    print(f"→ LP objective (continuous, binaries fixed) = {lp_obj:,.2f}\n")
    print("LP solve finished.\n")

    # 5) Print duals for your CO2 balance constraint
    #    (replace 'CO2_balance' and index set 'T' with whatever your builder uses)
    print("CO₂‐balance duals (only for fuel = 'CO2'):")
    for con in model.component_data_objects(Constraint, active=True):
        # Filter only the 'Balance' constraint family
        if con.parent_component().name == "Balance":
            a, e, t = con.index()        # unpack the (area, energy, time) tuple
            if e == "CO2":               # only for CO2
                π = model.dual[con]      # shadow price
                print(f"  area={a}, time={t}: dual = {π:,.4f}")

    # export_results_to_excel(model)
    print("Exporting results to Excel ...")
    export_results(model, cfg)
    print("Results exported successfully.")
    # debug_objective(model, cfg)

    elapsed = time.time() - start_time
    print("\n==========================")
    print("Pyomo Model Run Completed")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    elapsed_td = timedelta(seconds=int(elapsed))
    print(f"Total runtime: {elapsed_td}")

if __name__ == '__main__':
    main()
