# ============================================================
# src/utils/validate_strategic_model.py
# Comprehensive check of strategic-related model components
# ============================================================

from pyomo.environ import Param, Set

def validate_strategic_model(model, verbose=True):
    """
    Comprehensive validation of strategic-related model attributes.
    - Verifies existence and type of parameters a_co2, b_co2
    - Checks consistency of strategic supplier/demander sets
    - Prints structured report for debugging
    """
    print("\n[VALIDATION] --- Strategic Model Integrity Check ---")

    # --- Check 1: Inverse Demand Parameters ---
    for pname in ['a_co2', 'b_co2']:
        if not hasattr(model, pname):
            print(f"[ERROR] Missing parameter '{pname}' in model.")
            continue
        param = getattr(model, pname)
        if not isinstance(param, Param):
            print(f"[ERROR] '{pname}' exists but is not a Pyomo Param (type={type(param)}).")
            continue
        # Check numeric validity
        try:
            vals = [v for v in param.values()]
        except Exception:
            vals = []
        nvals = len(vals)
        if nvals == 0:
            print(f"[WARNING] Parameter '{pname}' contains no data.")
        else:
            numeric_ok = all(isinstance(v, (float, int)) for v in vals)
            print(f"[OK] {pname}: {nvals} entries, numeric={numeric_ok}, sample={vals[:3]}")

    # --- Check 2: Strategic actor sets ---
    for sname in ['StrategicSuppliers', 'StrategicDemanders']:
        if not hasattr(model, sname):
            print(f"[ERROR] Missing set '{sname}' in model.")
            continue
        sset = getattr(model, sname)
        if not isinstance(sset, Set):
            print(f"[ERROR] '{sname}' exists but is not a Pyomo Set (type={type(sset)}).")
            continue
        members = list(sset.data())
        print(f"[OK] {sname}: {len(members)} elements -> {members}")

        # Optional cross-check with model.G
        if hasattr(model, 'G'):
            invalid = [x for x in members if x not in model.G]
            if invalid:
                print(f"[WARNING] {sname} contains items not in model.G: {invalid}")

    # --- Cross-check consistency ---
    if hasattr(model, 'StrategicSuppliers') and hasattr(model, 'StrategicDemanders'):
        overlap = set(model.StrategicSuppliers.data()) & set(model.StrategicDemanders.data())
        if overlap:
            print(f"[WARNING] Overlap between suppliers and demanders: {overlap}")
        else:
            print("[OK] No overlap between StrategicSuppliers and StrategicDemanders.")

    print("[VALIDATION] --- Strategic Model Integrity Check Completed ---\n")

    # Optionally return a summary dict for programmatic tests
    summary = {
        'a_co2_exists': hasattr(model, 'a_co2'),
        'b_co2_exists': hasattr(model, 'b_co2'),
        'n_suppliers': len(model.StrategicSuppliers) if hasattr(model, 'StrategicSuppliers') else 0,
        'n_demanders': len(model.StrategicDemanders) if hasattr(model, 'StrategicDemanders') else 0,
    }
    return summary
