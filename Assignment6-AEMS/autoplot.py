
from pomdp import POMDP
from onlineSolver import OnlineSolver
from policyReader import PolicyReader
from aems import AEMS2
from mdpSolver import QMDP, MinMDP

import matplotlib.pyplot as plt

average_reward_rock_minmdp = []
average_reward_rock_pbvi = []
average_reward_tag_minmdp = []
average_reward_tag_pbvi = []

for model_name in ['RockSample_4_4', 'TagAvoid']:
    model_file = 'examples/env/' + model_name + '.pomdp'
    pomdp  = POMDP(model_file)
    precision = 0.1
    ub = QMDP(pomdp, precision)

    for lb_sover in ['MinMDP', 'PBVI']:
        if lb_sover == 'MinMDP':
            lb = MinMDP(pomdp, precision)
        elif lb_sover == 'PBVI':
            policy_file = 'examples/policy/' + model_name + '.policy'   # currently only for 3 problems
            lb = PolicyReader(pomdp, policy_file)

        for time_limit, num_runs in [(0.005, 500), (0.01, 500), (0.02, 500), (0.05, 500), (0.1, 500), (0.200, 200), (0.500, 100)]:
            print('=====================================')
            print(f'Start evaluation with time limits: {time_limit} second and {num_runs} runs with {lb_sover} as lower bound in model {model_name}')
            total_reward = 0
            for runs in range(num_runs):
                solver = AEMS2(pomdp, lb, ub, precision, time_limit)
                total_reward += OnlineSolver.solve(solver)
            average_reward = total_reward / num_runs
            print ('Average Reward: ', average_reward)
            if model_name == 'RockSample_4_4' and lb_sover == 'MinMDP':
                average_reward_rock_minmdp.append(average_reward)
            elif model_name == 'RockSample_4_4' and lb_sover == 'PBVI':
                average_reward_rock_pbvi.append(average_reward)
            elif model_name == 'TagAvoid' and lb_sover == 'MinMDP':
                average_reward_tag_minmdp.append(average_reward)
            elif model_name == 'TagAvoid' and lb_sover == 'PBVI':
                average_reward_tag_pbvi.append(average_reward)

x_data = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
for model_name in ['RockSample_4_4', 'TagAvoid']:
    print(f'### Plot the Average Reward of {model_name}')
    fig, ax = plt.subplots()
    ax.grid(True)

    if model_name == 'RockSample_4_4':
        plt.plot(x_data, average_reward_rock_minmdp, label = 'MinMDP')
        plt.plot(x_data, average_reward_rock_pbvi, label = 'PBVI')
    elif model_name == 'PBVI':
        plt.plot(x_data, average_reward_tag_minmdp, label = 'MinMDP')
        plt.plot(x_data, average_reward_tag_pbvi, label = 'PBVI')

    plt.xlabel('Time Limit')
    plt.ylabel('Average reward')
    plt.title(f'{model_name} Performance')
    plt.legend()

    print('Close the plot diagram to continue program')
    plt.show()
