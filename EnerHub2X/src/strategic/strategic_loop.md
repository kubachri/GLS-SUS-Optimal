**FUNCTION**        
***run_cournot***(cfg, tol, max_iter, damping, co2_label):

    # --- Initialization phase ---
    base_model ← build_model(cfg)
    solve(base_model)                          # centralized baseline
    strategies ← extract_current_sales(base_model, strategic_suppliers)
    print initial info / validate sets

    # --- Iteration phase (best-response loop) ---
    FOR iteration in 1..max_iter:
        max_change ← 0

        FOR each strategic_supplier i:
            # Build submodel for supplier i
            m ← build_model(cfg)
            fix_competitors_sales(m, strategies, excluding=i)

            # Solve submodel for supplier i by maximizing its profit
            define_strategic_objective(m)
            solve(m)

            # Update supplier i sales (to fix for other suppliers profit maximization)
            q_i_new ← extract_sales_for_supplier(m, i)
            q_i_updated ← damping * q_i_new + (1 - damping) * q_i_old
            strategies[i] ← q_i_updated

            # Compute maximal change in results to assess convergence
            max_change ← max(max_change, |q_i_updated - q_i_old|)

        PRINT iteration summary (max_change)
        IF max_change < tol: break (converged)

    # --- Finalization phase ---
    final_model ← build_model(cfg)
    fix_all_strategic_sales(final_model, strategies)
    solve(final_model)
    export_results()
    return final_model, strategies
