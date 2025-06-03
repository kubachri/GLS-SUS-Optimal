# scripts/run_model.py
import argparse
from pyomo.environ import SolverFactory, Suffix
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

    model.dual = Suffix(direction=Suffix.IMPORT)

    solver = SolverFactory('gurobi')
    solver.options['MIPGap'] = 0.0015
    solver.solve(model, tee=True)

    print("\nWeekly methanol shadow prices (€/t or utility units per tonne):")
    for w in model.W:
        constr = model.WeeklyMethanolTarget[w]
        dual = model.dual.get(constr)
        print(f"Week {w:>2}:  {dual:8.2f} €/t" if dual is not None else f"Week {w:>2}:  (inactive or 0)")

    # export_results_to_excel(model)
    export_results(model)

if __name__ == '__main__':
    main()
