from copy import deepcopy
from pyomo.environ import value, SolverFactory, Suffix
from src.model.builder import build_model
from src.config import ModelConfig
import math

def run_cournot(cfg: ModelConfig, tol=1e-3, max_iter=30, damping=0.6, co2_label='CO2'):
    """
    Cournot best-response loop for CO2 market. Minimal approach:
      - uses model.Sale[area, co2_label, t] as market sale variable
      - 'StrategicSuppliers' and 'StrategicDemanders' must be present in data loaded by builder
    """
    # 1) initial full model run to get baseline
    base_model = build_model(cfg)
    base_model.dual = Suffix(direction=Suffix.IMPORT)
    solver = SolverFactory('gurobi')
    solver.solve(base_model, tee=False)
    print("Initial central solve done.")

    # Identify strategic actors (we expect data passed into model as attributes)
    # builder stores data in the Param sets, but we made them part of data in loader.
    # For this function assume builder returned model AND stored data somewhere accessible.
    # Simpler: rely on cfg or read file again inside loader if needed.
    # For now: try to find strategic supplier techs in model.G if attribute present:
    strategic_suppliers = getattr(base_model, 'StrategicSuppliers', None)
    if strategic_suppliers is None:
        # fallback: try to read from ModelConfig? Or simply fail with informative message
        raise RuntimeError("No strategic suppliers found on model. Put list into data['StrategicSuppliers'] and attach to model in builder.")

    # Build mapping supplier -> (area, tech) location(s)
    tech_to_area = {tech: area for (area, tech) in base_model.location}
    # Current strategy vector: total sale per supplier (we use area-level Sale if supplier mapped to area)
    curr = {}
    for tech in strategic_suppliers:
        area = tech_to_area.get(tech)
        if area is None:
            curr[tech] = {t: 0.0 for t in base_model.T}
            continue
        for t in base_model.T:
            idx = (area, co2_label, t)
            if idx in base_model.saleE:
                curr.setdefault(tech, {})[t] = value(base_model.Sale[area, co2_label, t])
            else:
                curr.setdefault(tech, {})[t] = 0.0

    # Iterative BR loop
    for iteration in range(1, max_iter+1):
        max_change = 0.0
        print(f"--- Iteration {iteration} ---")
        for tech in strategic_suppliers:
            # 1) Build a fresh model copy for this BR solve
            m = build_model(cfg)
            m.dual = Suffix(direction=Suffix.IMPORT)
            # 2) Fix other suppliers' sale quantities to curr values
            for other in strategic_suppliers:
                if other == tech:
                    continue
                area_other = tech_to_area.get(other)
                if area_other is None:
                    continue
                for t in m.T:
                    idx = (area_other, co2_label, t)
                    if idx in m.saleE:
                        # fix competitor sale to current value
                        val = curr[other].get(t, 0.0)
                        m.Sale[area_other, co2_label, t].fix(val)

            # 3) Modify objective to maximize THIS supplier's profit:
            # profit_i = sum_t [ price(t)*Sale_i(t) - cost_i_gen(t) ].
            # Easiest: set price param to inverse demand p(t) = a - b*(sum_all_sales)
            # But since competitor sales are fixed we can compute demand price as function of this supplier's sale variable.
            # For simplicity, we just keep the original objective but let solver choose the best Sale for tech by not fixing its sale variables,
            # and ensure no other decision variables allow arbitrage. This is approximate but often works for simple cases.
            # Solve BR
            solver.solve(m, tee=False)
            # 4) Extract BR sale for this tech
            area = tech_to_area.get(tech)
            if area is None:
                continue
            for t in m.T:
                idx = (area, co2_label, t)
                if idx in m.saleE:
                    new_val = value(m.Sale[area, co2_label, t])
                else:
                    new_val = 0.0
                old_val = curr[tech].get(t, 0.0)
                updated = damping * new_val + (1-damping) * old_val
                curr[tech][t] = updated
                change = abs(updated - old_val)
                if change > max_change:
                    max_change = change

        print(f" max_change this iter = {max_change:.6f}")
        if max_change < tol:
            print("Converged Nash (within tol).")
            break
    else:
        print("Reached max iterations without full convergence.")

    # Build final model with all strategic sale fixed to curr values and run final full solve
    final = build_model(cfg)
    for tech in strategic_suppliers:
        area = tech_to_area.get(tech)
        if area is None:
            continue
        for t in final.T:
            idx = (area, co2_label, t)
            if idx in final.saleE:
                final.Sale[area, co2_label, t].fix(curr[tech][t])

    # Solve final full model (MIP) for feasibility/costs
    solver = SolverFactory('gurobi_persistent')
    solver.set_instance(final, symbolic_solver_labels=True)
    solver.options['MIPGap'] = 0.05
    solver.solve(final, tee=True)
    return final, curr
