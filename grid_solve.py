from pulp import *
import pulp
import numpy as np

def solve_random_lp(T, num_vehicles, speed, num_stations, num_chargers_per_station, max_rate_of_charge, battery_per_mile, grid_size):
    D = 2
    size = grid_size

    # generate random start and end locations
    vehicle_start = np.random.randint(0, size, (num_vehicles, 2))
    vehicle_end = np.random.randint(0, size, (num_vehicles, 2))

    # charger_loc = [[0,0] for i in range(num_stations)]
    charger_loc = np.random.randint(0, size, (num_stations, 2))
    charger_co2s = [0.1 for i in range(num_stations)]

    starting_battery = [100 for i in vehicle_start]
    ending_battery = [20 for i in vehicle_end]

    # create a linear program
    prob = LpProblem("CVRP with Charging Stations", LpMinimize)

    # convert x into a lp variable
    x = LpVariable.dicts("x",[(i, j, k) for i in range(len(vehicle_start)) for j in range(T) for k in range(D)], lowBound=0, upBound=size, cat=LpInteger)

    # variable for tracking the current charge level
    b = LpVariable.dicts("b",[(i, j) for i in range(len(vehicle_start)) for j in range(T)], lowBound=0, upBound=100, cat=LpInteger)

    # variable for holding the distance traveled
    d = LpVariable.dicts("d",[(i, j, k) for i in range(len(vehicle_start)) for j in range(T-1) for k in range(D)], lowBound=0, upBound=None, cat=LpInteger)

    # variable for holding the how much charge is added
    c = LpVariable.dicts("c",[(i, ci, t) for ci in range(len(charger_loc)) for i in range(len(vehicle_start)) for t in range(T-1)], lowBound=0, upBound=max_rate_of_charge, cat=LpInteger)
    
    # binary variable for tracking whether c is positive
    c_cond = LpVariable.dicts("c_cond",[(i, ci, t) for ci in range(len(charger_loc)) for i in range(len(vehicle_start)) for t in range(T-1)], lowBound=0, upBound=1, cat=LpBinary)

    # absolute variable for difference between cars and charging stations
    c_diff = LpVariable.dicts("c_diff",[(i, ci, t, j) for j in range(2) for ci in range(len(charger_loc)) for i in range(len(vehicle_start)) for t in range(T-1)], lowBound=0, upBound=None, cat=LpInteger)
    """
    Model charging for the battery. We want to enforce the constraint such that when a car is close enough to one of the charging locations, then it can get charged there. 
    Each car should only be able to connect to one charger and one charger should only be able to serve up to num_chargers_per_station cars
    """

    # cars start with a specific battery level
    for i in range(len(vehicle_start)):
        prob += b[i, 0] == starting_battery[i]

    # cars end with at least a specific battery level
    for i in range(len(vehicle_start)):
        prob += b[i, T-1] >= ending_battery[i]

    # battery must be above 10 at all times
    for i in range(len(vehicle_start)):
        for t in range(T):
            prob += b[i, t] >= 10

    for i in range(len(vehicle_start)):
        for ci in range(len(charger_loc)):
            for t in range(T-1):
                c_diff[i, ci, t, 0] >= (x[i, t, 0] - charger_loc[ci][0])
                c_diff[i, ci, t, 0] >= -(x[i, t, 0] - charger_loc[ci][0])

                c_diff[i, ci, t, 1] >= (x[i, t, 1] - charger_loc[ci][1])
                c_diff[i, ci, t, 1] >= -(x[i, t, 1] - charger_loc[ci][1])

                c_cond[i, ci, t] = (c_diff[i, ci, t, 0] + c_diff[i, ci, t, 1]) == 0
                c[i, ci, t] <= c_cond[i, ci, t] - 1
                # if the car is close enough to the charger, then it can be charged
                # prob += c_cond[i, ci, t] >= (x[i, t, 0] - charger_loc[ci][0]) * (x[i, t, 0] - charger_loc[ci][0]) + (x[i, t, 1] - charger_loc[ci][1]) * (x[i, t, 1] - charger_loc[ci][1])

    obj1 = 0
    time_objective = 0
    # one charger can only serve num_chargers_per_station cars
    for ci in range(len(charger_loc)):
        for t in range(T-1):
            charger_consumption = lpSum(c[i, ci, t] for i in range(len(vehicle_start)))
            obj1 += charger_consumption * charger_co2s[ci]
            prob += charger_consumption <= num_chargers_per_station * max_rate_of_charge

    # only one car can be connected to a charger
    for i in range(len(vehicle_start)):
        for t in range(T-1):
            prob += lpSum(c[i, ci, t] for ci in range(len(charger_loc))) <= max_rate_of_charge

    # enforce a speed constraint for lp variables
    for i in range(len(vehicle_start)):
        for t in range(T-1):
            prob += x[i, t+1, 1] - x[i, t, 1] <= d[i, t, 1]
            prob += x[i, t, 1] - x[i, t+1, 1] <= d[i, t, 1]

            prob += x[i, t+1, 0] - x[i, t, 0] <= d[i, t, 0]
            prob += x[i, t, 0] - x[i, t+1, 0] <= d[i, t, 0]

            prob += d[i, t, 0] + d[i, t, 1] <= speed
            time_objective += (d[i, t, 0] + d[i, t, 1]) * t
            prob += b[i, t+1] == b[i, t] + lpSum(c[i, ci, t] for ci in range(len(charger_loc))) - (d[i, t, 0] + d[i, t, 1]) * battery_per_mile

    # enforce the final location constraint
    for i in range(len(vehicle_start)):
        prob += x[i, T-1, 0] == vehicle_end[i, 0]
        prob += x[i, T-1, 1] == vehicle_end[i, 1]

    # enforce the start location constraint
    for i in range(len(vehicle_start)):
        prob += x[i, 0, 0] == vehicle_start[i, 0]
        prob += x[i, 0, 1] == vehicle_start[i, 1]

    obj2 = lpSum(lpSum(lpSum(d[i, t-1, k] for k in range(2)) for t in range(1, T)) for i in range(len(vehicle_start)))

    prob += obj1 + obj2 + time_objective

    prob.solve()
    x_out = np.zeros((len(vehicle_start), T, D))
    for i in range(len(vehicle_start)):
        for t in range(T):
            for k in range(D):
                x_out[i,t, k] = x[i,t, k].varValue

    # create c out
    c_out = np.zeros((len(vehicle_start), len(charger_loc), T-1))
    for i in range(len(vehicle_start)):
        for ci in range(len(charger_loc)):
            for t in range(T-1):
                c_out[i, ci, t] = c[i, ci, t].varValue


    # create b out
    b_out = np.zeros((len(vehicle_start), T))
    for i in range(len(vehicle_start)):
        for t in range(T):
            b_out[i, t] = b[i, t].varValue

    objective = pulp.value(prob.objective)
    print("Objective = {}".format(objective))

    return prob, objective, x_out, c_out, b_out, vehicle_start, vehicle_end
    

def solve_lp(T, vehicle_start, vehicle_end, charger_loc, charger_co2s, starting_battery, ending_battery, speed, num_chargers_per_station, max_rate_of_charge, battery_per_mile, grid_size):
    D = 2
    size = grid_size

    # create a linear program
    prob = LpProblem("CVRP with Charging Stations", LpMinimize)

    # convert x into a lp variable
    x = LpVariable.dicts("x",[(i, j, k) for i in range(len(vehicle_start)) for j in range(T) for k in range(D)], lowBound=0, upBound=size, cat=LpInteger)

    # variable for tracking the current charge level
    b = LpVariable.dicts("b",[(i, j) for i in range(len(vehicle_start)) for j in range(T)], lowBound=0, upBound=100, cat=LpInteger)

    # variable for holding the distance traveled
    d = LpVariable.dicts("d",[(i, j, k) for i in range(len(vehicle_start)) for j in range(T-1) for k in range(D)], lowBound=0, upBound=None, cat=LpInteger)

    # variable for holding the how much charge is added
    c = LpVariable.dicts("c",[(i, ci, t) for ci in range(len(charger_loc)) for i in range(len(vehicle_start)) for t in range(T-1)], lowBound=0, upBound=max_rate_of_charge, cat=LpInteger)
    
    # binary variable for tracking whether c is positive
    c_cond = LpVariable.dicts("c_cond",[(i, ci, t) for ci in range(len(charger_loc)) for i in range(len(vehicle_start)) for t in range(T-1)], lowBound=0, upBound=1, cat=LpBinary)

    # absolute variable for difference between cars and charging stations
    c_diff = LpVariable.dicts("c_diff",[(i, ci, t, j) for j in range(2) for ci in range(len(charger_loc)) for i in range(len(vehicle_start)) for t in range(T-1)], lowBound=0, upBound=None, cat=LpInteger)
    """
    Model charging for the battery. We want to enforce the constraint such that when a car is close enough to one of the charging locations, then it can get charged there. 
    Each car should only be able to connect to one charger and one charger should only be able to serve up to num_chargers_per_station cars
    """

    # cars start with a specific battery level
    for i in range(len(vehicle_start)):
        prob += b[i, 0] == starting_battery[i]

    # cars end with at least a specific battery level
    for i in range(len(vehicle_start)):
        prob += b[i, T-1] >= ending_battery[i]

    # battery must be above 10 at all times
    for i in range(len(vehicle_start)):
        for t in range(T):
            prob += b[i, t] >= 10

    for i in range(len(vehicle_start)):
        for ci in range(len(charger_loc)):
            for t in range(T-1):
                c_diff[i, ci, t, 0] >= (x[i, t, 0] - charger_loc[ci][0])
                c_diff[i, ci, t, 0] >= -(x[i, t, 0] - charger_loc[ci][0])

                c_diff[i, ci, t, 1] >= (x[i, t, 1] - charger_loc[ci][1])
                c_diff[i, ci, t, 1] >= -(x[i, t, 1] - charger_loc[ci][1])

                # c_cond[i, ci, t] = (c_diff[i, ci, t, 0] + c_diff[i, ci, t, 1]) == 0
                # c[i, ci, t] <= (c_cond[i, ci, t] - 1) * max_rate_of_charge
                # if the car is close enough to the charger, then it can be charged
                # prob += c_cond[i, ci, t] >= (x[i, t, 0] - charger_loc[ci][0]) * (x[i, t, 0] - charger_loc[ci][0]) + (x[i, t, 1] - charger_loc[ci][1]) * (x[i, t, 1] - charger_loc[ci][1])

    obj1 = 0
    time_objective = 0
    # one charger can only serve num_chargers_per_station cars
    for ci in range(len(charger_loc)):
        for t in range(T-1):
            charger_consumption = lpSum(c[i, ci, t] for i in range(len(vehicle_start)))
            obj1 += charger_consumption * charger_co2s[ci]
            prob += charger_consumption <= num_chargers_per_station * max_rate_of_charge

    # only one car can be connected to a charger
    for i in range(len(vehicle_start)):
        for t in range(T-1):
            prob += lpSum(c[i, ci, t] for ci in range(len(charger_loc))) <= max_rate_of_charge

    # enforce a speed constraint for lp variables
    for i in range(len(vehicle_start)):
        for t in range(T-1):
            prob += x[i, t+1, 1] - x[i, t, 1] <= d[i, t, 1]
            prob += x[i, t, 1] - x[i, t+1, 1] <= d[i, t, 1]

            prob += x[i, t+1, 0] - x[i, t, 0] <= d[i, t, 0]
            prob += x[i, t, 0] - x[i, t+1, 0] <= d[i, t, 0]

            prob += d[i, t, 0] + d[i, t, 1] <= speed
            time_objective += (d[i, t, 0] + d[i, t, 1]) * t
            prob += b[i, t+1] == b[i, t] + lpSum(c[i, ci, t] for ci in range(len(charger_loc))) - (d[i, t, 0] + d[i, t, 1]) * battery_per_mile

    # enforce the final location constraint
    for i in range(len(vehicle_start)):
        prob += x[i, T-1, 0] == vehicle_end[i, 0]
        prob += x[i, T-1, 1] == vehicle_end[i, 1]

    # enforce the start location constraint
    for i in range(len(vehicle_start)):
        prob += x[i, 0, 0] == vehicle_start[i, 0]
        prob += x[i, 0, 1] == vehicle_start[i, 1]

    obj2 = lpSum(lpSum(lpSum(d[i, t-1, k] for k in range(2)) for t in range(1, T)) for i in range(len(vehicle_start)))

    prob += obj1 + obj2 + time_objective

    prob.solve()
    x_out = np.zeros((len(vehicle_start), T, D))
    for i in range(len(vehicle_start)):
        for t in range(T):
            for k in range(D):
                x_out[i,t, k] = x[i,t, k].varValue

    # create c out
    c_out = np.zeros((len(vehicle_start), len(charger_loc), T-1))
    for i in range(len(vehicle_start)):
        for ci in range(len(charger_loc)):
            for t in range(T-1):
                c_out[i, ci, t] = c[i, ci, t].varValue


    # create b out
    b_out = np.zeros((len(vehicle_start), T))
    for i in range(len(vehicle_start)):
        for t in range(T):
            b_out[i, t] = b[i, t].varValue

    # create c cond out
    c_cond_out = np.zeros((len(vehicle_start), len(charger_loc), T-1))
    for i in range(len(vehicle_start)):
        for ci in range(len(charger_loc)):
            for t in range(T-1):
                c_cond_out[i, ci, t] = c_cond[i, ci, t].value()

    objective = pulp.value(prob.objective)
    print("Objective = {}".format(objective))

    return prob, objective, x_out, c_out, b_out, c_cond_out, vehicle_start, vehicle_end
    
