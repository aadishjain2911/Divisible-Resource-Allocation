from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
import numpy as np
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, LpMinimize, LpConstraintVar, LpConstraint, PULP_CBC_CMD
import matplotlib.pyplot as plt
from numpy import arange
from tqdm import tqdm

num_agents = 4
arrivals = [0, 2, 5, 3]
departures = [3, 5, 10, 7]
peaks = [2, 4, 6, 3]
epsilon = 0.0001
MAX = 100000
PRINT = True

def get_intervals(num_agents, arrivals, departures, peaks):
    intervals = list(set(arrivals + departures))
    intervals.append(0)
    intervals.sort()
    interval_lengths = []
    for i in range(len(intervals) - 1):
        interval_lengths.append(intervals[i + 1] - intervals[i])
    connected_agents = [[] for i in range(len(interval_lengths))]
    for i in range(num_agents):
        for j in range(len(interval_lengths)):
            if arrivals[i] <= intervals[j] and departures[i] >= intervals[j + 1]:
                connected_agents[j].append(i)
    return intervals, interval_lengths, connected_agents

def generate_random_theta(num_agents):
    arrivals = []
    departures = []
    peaks = []
    for i in range(num_agents):
        arrivals.append(np.random.randint(0, 19))
        departures.append(np.random.randint(arrivals[i]+1, 20))
        peaks.append(np.random.randint(1, 10))
    return arrivals, departures, peaks, get_intervals(num_agents, arrivals, departures, peaks)

intervals, interval_lengths, connected_agents = get_intervals(num_agents, arrivals, departures, peaks)

for _ in tqdm(range(1)):
    # arrivals, departures, peaks, (intervals, interval_lengths, connected_agents) = generate_random_theta(num_agents)

    if PRINT:
        print("Peaks: ", peaks)
        print("Interval lengths: ", interval_lengths)
        print("Connected agents: ", connected_agents)
        print("Arrivals: ", arrivals)
        print("Departures: ", departures)

    # Create the directed graph
    # The first node is the source, the last node is the sink
    # The nodes in between are the agents and the intervals
    num_intervals = len(interval_lengths)
    num_nodes = num_agents + num_intervals + 2

    # create a new csr_matrix
    # the matrix is num_nodes x num_nodes
    graph = csr_matrix((num_nodes, num_nodes), dtype=np.int32)

    for i in range(num_intervals):
        # connect the source to the interval
        graph[0, i + 1] = interval_lengths[i]

        for j in connected_agents[i]:
            # connect the interval to the agent
            graph[i + 1, j + num_intervals + 1] = peaks[j]

    for i in range(num_agents):
        # connect the agent to the sink
        graph[i + num_intervals + 1, -1] = peaks[i]


    # Solve the maximum flow problem
    flow_result = maximum_flow(graph, 0, num_nodes - 1)
    total_flow = 0
    for i in range(num_intervals):
        total_flow += flow_result.flow[0, i + 1]

    if PRINT:
        print("Total flow: ", total_flow)
        print("Agent allocations:")
        for i in range(num_agents):
            print(f"Agent {i}: {flow_result.flow[i + num_intervals + 1, -1]}")

        print("Interval allocations:")
        for i in range(num_intervals):
            print(f"Interval {i}: {flow_result.flow[0, i + 1]}")


    # LP construction
    # for every tuple in connected_agents there should be a variable

    # create the variables
    variables = []
    for i in range(num_intervals):
        variables.append([])
        for j in range(num_agents):
            variables[i].append(LpVariable(name=f"alloc_{i}_{j}", lowBound=0))

    # value_vars[i][k] = min(value_k, peaks[i])
    value_vars = []
    for i in range(num_agents):
        value_vars.append([])
        for j in range(num_agents):
            value_vars[i].append(LpVariable(name=f"value_{i}_{j}", lowBound=0))

    # create the model
    model = LpProblem(name="flow", sense=LpMaximize)

    # add the objective function
    # model.setObjective(lpSum([value_vars[i][j] for i in range(num_agents) for j in range(num_agents)]))
    model.setObjective(lpSum([variables[i][j] for i in range(num_intervals) for j in range(num_agents)]))

    # add the constraints

    # presence constraint
    for i in range(num_intervals):
        for j in range(num_agents):
            if j not in connected_agents[i]:
                model.addConstraint(variables[i][j] == 0)

    # peaks constraint
    for i in range(num_agents):
        value = 0
        for j in range(num_intervals):
            if i in connected_agents[j]:
                value += variables[j][i]
        model.addConstraint(value <= peaks[i])

    # interval constraint
    for i in range(num_intervals):
        model.addConstraint(lpSum(variables[i]) <= interval_lengths[i])

    # pareto optimality
    # model.addConstraint(lpSum([variables[i][j] for i in range(num_intervals) for j in range(num_agents)]) == total_flow)

    # model.addConstraint(variables[1][0] == variables[1][1])
    # model.addConstraint(variables[1][1] == variables[1][2])

    # EF-A
    for i in range(num_agents):
        value = 0
        for j in range(num_intervals):
            if i in connected_agents[j]:
                value += variables[j][i]

        # determine agents that are one-hop away from agent i
        one_hop = set()
        for j in range(num_intervals):
            if i in connected_agents[j]:
                one_hop = one_hop.union(connected_agents[j])

        for k in one_hop:
            value_k = 0
            for j in range(num_intervals):
                if k in connected_agents[j] and i in connected_agents[j]:
                    value_k += variables[j][k]

            delta = LpVariable(name=f"delta_{i}_{k}", cat='Binary')

            model.addConstraint(value >= value_vars[i][k])
            model.addConstraint(value_vars[i][k] <= value_k)
            model.addConstraint(value_vars[i][k] <= peaks[i])
            model.addConstraint(value_vars[i][k] >= value_k - (1-delta)*(MAX))
            model.addConstraint(value_vars[i][k] >= peaks[i] - (delta)*(MAX))

    # solve the model
    status = model.solve(PULP_CBC_CMD(msg=False))
    assert abs(model.objective.value() - total_flow) <= epsilon
    assert LpStatus[status] == "Optimal"
    for i in range(num_agents):
        value = 0
        for j in range(num_intervals):
            if i in connected_agents[j]:
                value += variables[j][i].value()

        # determine agents that are one-hop away from agent i
        one_hop = set()
        for j in range(num_intervals):
            if i in connected_agents[j]:
                one_hop = one_hop.union(connected_agents[j])

        for k in one_hop:
            value_k = 0
            for j in range(num_intervals):
                if k in connected_agents[j] and i in connected_agents[j]:
                    value_k += variables[j][k].value()

            assert abs(value_vars[i][k].value() - value_k) <= epsilon or abs(value_vars[i][k].value() - peaks[i]) <= epsilon

    if PRINT:
        # print the results
        print("Status:", LpStatus[status])
        print("Objective value:", model.objective.value())

        # print agent wise allocations
        for i in range(num_agents):
            total = 0
            for j in range(num_intervals):
                if i in connected_agents[j]:
                    total += variables[j][i].value()
            print(f"Agent {i}: {total}")

        for i in range(num_intervals):
            for j in range(num_agents):
                print(f"Interval {i}, agent {j}: {variables[i][j].value()}")

    # plot the results
    fig, gnt = plt.subplots()
    gnt.set_title("Flow allocation")
    gnt.set_ylim(0, np.sum(interval_lengths)+1)
    gnt.set_xlim(0, num_agents+1)
    gnt.set_xlabel('peaks')
    gnt.set_ylabel('time')
    gnt.set_xticks(arange(1, num_agents+1, 1))
    gnt.set_xticklabels(peaks)
    gnt.grid(True)

    start = 0
    for i, row in enumerate(variables):
        offset = 0
        for j, variable in enumerate(row):
            gnt.broken_barh([(j+1, 0.2)], (start+offset, variable.value()), facecolors =('tab:orange'))
            offset += variable.value()
        start += interval_lengths[i]

    # mark the arrival and departure times with a line parallel to x-axis
    for i in range(num_agents):
        gnt.broken_barh([(i+1, 0.2)], (arrivals[i], 0.1), facecolors =('tab:blue'))
        gnt.broken_barh([(i+1, 0.2)], (departures[i], 0.1), facecolors =('tab:red'))

    plt.savefig("ev_alloc_flow.png")

    plt.close()

