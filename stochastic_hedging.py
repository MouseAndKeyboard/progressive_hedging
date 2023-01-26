import gurobipy as gp
import numpy as np

plants = ['wheat', 'rice', 'corn']
scenarios = ['good', 'bad']

plant_prices = {
    'wheat': 1.,
    'rice': 0.5,
    'corn': 2.,
}

scenario_probabilities = {
    'good': 0.5,
    'bad': 0.5,
}

# Yield is a multiplier of how much was initially planted
plant_yield = {
    'wheat': {
        'good': 2.,
        'bad': 1.5,
    },
    'rice': {
        'good': 3.,
        'bad': 1.,
    },
    'corn': {
        'good': 2.,
        'bad': 0.5,
    }
}

sell_prices = {
    'wheat': {
        'good': 2.,
        'bad': 2.,
    },
    'rice': {
        'good': 1.5,
        'bad': 1.5,
    },
    'corn': {
        'good': 4.,
        'bad': 4.,
    }
}

market_restriction = {
    'wheat': {
        'good': 50.,
        'bad': 100.,
    },
    'rice': {
        'good': 80.,
        'bad': 80.,
    },
    'corn': {
        'good': 30.,
        'bad': 30.,
    }
}

initial_money = 30.


S = len(scenarios)
P = len(plants)

def get_c_vector():
    c = np.zeros(P)
    for i, plant in enumerate(plants):
        c[i] = plant_prices[plant]
    return c

def get_q_matrix():
    q = np.zeros((S, P))
    for i, scenario in enumerate(scenarios):
        for j, plant in enumerate(plants):
            q[i][j] = sell_prices[plant][scenario]
    return q

def get_p_vector():
    p = np.zeros(S)
    for i, scenario in enumerate(scenarios):
        p[i] = scenario_probabilities[scenario]
    return p

buy_prices = get_c_vector()
sell_prices = get_q_matrix()

def Lagrangian_s(rho, buy_prices, sell_prices, multiplier, plant_qty, sell_qty, prev_avg, magic=False):
    if magic:
        return inner(buy_prices + multiplier, plant_qty) - \
           inner(sell_prices, sell_qty)


    return inner(buy_prices + multiplier, plant_qty) - \
           inner(sell_prices, sell_qty) + \
           inner(multiplier, plant_qty - prev_avg) + \
           (rho/2) * inner(plant_qty - prev_avg, plant_qty - prev_avg, P)

def inner(a, b, length=None):
    if length is None:
        return sum(a[i] * b[i] for i in range(len(a)))
    else:
        return sum(a[i] * b[i] for i in range(length))

def subproblem(s, rho, prev_avg_planted, multiplier, magic=False):
    """
        Solves the argmin:
        (c + w)x + q y | (x, y) in K_s

    """
    model = gp.Model("subproblem")
    x = model.addMVar(P, lb=0, name="qty_planted")
    y = model.addMVar(P, lb=0, name="qty_sold")

    model.setObjective(Lagrangian_s(rho, buy_prices, sell_prices[s], multiplier[s], x, y, prev_avg_planted, magic), sense=gp.GRB.MINIMIZE)

    model.addConstrs(
        ( inner(buy_prices, x) <= initial_money for _ in [1] ),
        name = "initial_money"
    )

    model.addConstrs(
        ( y[i] <= plant_yield[plant][scenarios[s]] * x[i] for i, plant in enumerate(plants) ),
        name = "plant_yield"
    )

    model.addConstrs(
        ( y[i] <= market_restriction[plant][scenarios[s]] for i, plant in enumerate(plants) ),
        name = "market_restriction"
    )
    
    #We write the problem for Gurobi
    model.write('farmer_progressive.lp')

    #We tell Gurobi to run quietly (i.e. don't print out stuff)
    model.Params.LogToConsole = 0
    #We tell Gurobi to solve the QP subproblem
    model.optimize()

    mult = model.pi

    return x.X, y.X, mult

def progressive_hedge(rho, k_max, initial_multiplier):
    qty_planted = np.zeros((S, P)) 
    qty_sold = np.zeros((S, P))
    p = get_p_vector()

    for s, scenario in enumerate(scenarios):
        planted, sold, sub_multipliers = subproblem(s, rho, None, initial_multiplier, magic=True)
        qty_planted[s] = planted
        qty_sold[s] = sold

    prev_avg_planted = sum(p[s] * qty_planted[s] for s in range(S))

    multiplier = np.zeros((S, P))
    for s in range(S):
        multiplier[s] = initial_multiplier[s] + rho * (qty_planted[s] - prev_avg_planted)

    for k in range(k_max):
        
        # x update
        for s, scenario in enumerate(scenarios):
            planted, sold, sub_multipliers = subproblem(s, rho, prev_avg_planted, multiplier)
            qty_planted[s] = planted
            qty_sold[s] = sold

        prev_avg_planted = sum(p[s] * qty_planted[s] for s in range(S))
        
        for s in range(S):
            multiplier[s] = multiplier[s] + rho * (qty_planted[s] - prev_avg_planted)

    return qty_planted, qty_sold, sub_multipliers

if __name__ == "__main__":
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    qty_planted, qty_sold, mults = progressive_hedge(1, 100, np.zeros((S, P)))
    
    print(f"Scenario Probabilities: " + ", ".join(f"{s}: {p}" for s, p in scenario_probabilities.items()))
    for i, plant in enumerate(plants):
        print(f"{plant} planted: {round(qty_planted[0][i], 3)}")
        for s, scenario in enumerate(scenarios):
            print(f"\tWhen {scenario} sold: {round(qty_sold[s][i], 3)}")

    for s, scenario in enumerate(scenarios):
        print(f"When {scenario} total profit: " + str(sum(qty_sold[s][i] * sell_prices[s][i] - qty_planted[s][i] * buy_prices[i] for i in range(P))))