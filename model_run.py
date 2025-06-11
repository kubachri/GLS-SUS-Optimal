# scripts/run_model.py
import argparse
from pyomo.environ import SolverFactory, Suffix, value, Var, Binary
from src.config           import ModelConfig
from src.model.builder    import build_model
from src.data.loader import load_data
from src.utils.excel_writer import export_results_to_excel
from src.utils.export_resultT import export_results

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test', action='store_true',
                   help="short horizon test run")
    p.add_argument('--n-test', type=int, default=168,
                   help="hours to keep when --test is on")
    p.add_argument('--penalty', type=float, default=1e6,
                   help="penalty multiplier for slack in objective")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = ModelConfig(test_mode=args.test,
                      n_test=args.n_test,
                      penalty=args.penalty)

    model = build_model(cfg)

    if model.Demand_Target:
        # 2) Tell Pyomo we want to import duals (for later LP)
        model.dual = Suffix(direction=Suffix.IMPORT)

    # 3) Solve the MIP
    solver = SolverFactory('gurobi')
    solver.options['MIPGap'] = 0.0015
    print("Solving MIP …")
    mip_result = solver.solve(model, tee=True)
    print("MIP solve finished.\n")

    if model.Demand_Target:

        # After solving the MIP, but before fixing binaries:
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
        model.dual.clear()
        print("Re‐solving as an LP to extract duals …")
        lp_result = solver.solve(model, tee=False)
        print("LP solve finished.\n")

    # export_results_to_excel(model)
    export_results(model)

if __name__ == '__main__':
    main()
