# src/model/objective.py

from pyomo.environ import Constraint, Objective, maximize

def define_objective(m, penalty=1e6):
    # ProfitDefinition handled in a constraint

    def profit_definition_rule(m):
        # a) Fuel cost (imports are a positive cost → negative in objective)
        imp_cost = sum(
            m.price_buy[a,e,t] * m.Buy[a,e,t]
            for (a,e) in m.buyE
            for t in m.T
        )
        # b) Sale revenue
        sale_rev = sum(
            m.price_sale[a,e,t] * m.Sale[a,e,t]
            for (a,e) in m.saleE
            for t in m.T
        )
        # c) Variable O&M on all tech→energy links
        var_om = sum(
            m.Generation[g,e,t] * m.cvar[g]
            for (g,e) in m.TechToEnergy
            for t in m.T
        )
        # d) Startup costs
        startup = sum(
            m.Startcost[g,t]
            for g in m.G
            for t in m.T
        )
        # e) Slack penalties (both import‐slack and export‐slack)
        slack_imp_sum = sum(
            m.SlackDemandImport[a, e, t]
            for (a, e, t) in m.DemandSet
            if (a, e) in m.buyE
        )
        slack_exp_sum = sum(
            m.SlackDemandExport[a, e, t]
            for (a, e, t) in m.DemandSet
            if (a, e) in m.saleE
        )

        # GAMS: Cost =E= -imp_cost + sale_rev - var_om - startup - penalty*slack
        return m.Profit == (
           - imp_cost
           + sale_rev
           - var_om
           - startup
           - penalty * (slack_imp_sum + slack_exp_sum)
        )

    #Impose constraint (profit definition)
    m.ProfitDefinition = Constraint(rule=profit_definition_rule)

    #Define objective function as profit maximization
    m.Obj = Objective(expr=m.Profit, sense=maximize)