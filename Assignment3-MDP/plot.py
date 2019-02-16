import sys, time

import matplotlib.pyplot as plt

def get_average_reward(agent, env):
    displayCallback = lambda x: None
    messageCallback = lambda x: None
    pauseCallback = lambda : None
    decisionCallback = a.getAction

    print(f'Running {episodes_count} Episodes')
    returns = 0
    for episode in range(1, episodes_count + 1):
        returns += gridworld.runEpisode(agent, env, discount, decisionCallback, displayCallback, messageCallback, pauseCallback, episode)
    average_reward = (returns + 0.0) / episodes_count
    print(f'Average Returns from Start State: {average_reward}')
    return average_reward

if __name__ == '__main__':
    episodes_count = 1000
    discount = 0.9

    ###########################
    # GET THE GRIDWORLD
    ###########################

    import gridworld
    mdpFunction = getattr(gridworld, "getBigGrid")
    mdp = mdpFunction()
    mdp.setLivingReward(0)
    mdp.setNoise(0.2)
    env = gridworld.GridworldEnvironment(mdp)

    ###########################
    # GET THE DISPLAY ADAPTER
    ###########################

    import textGridworldDisplay
    display = textGridworldDisplay.TextGridworldDisplay(mdp)
    try:
        display.start()
    except KeyboardInterrupt:
        sys.exit(0)

    ###########################
    # GET THE AGENT
    ###########################

    import valueIterationAgents, rtdpAgents

    vi_time = []
    vi_average_reward = []
    rtdp_time = []
    rtdp_average_reward = []

    print('==================== Value Iteration ====================')
    for iteration in range(20):
        startTime = time.time()
        a = valueIterationAgents.ValueIterationAgent(mdp, discount, iteration)

        planning_time = time.time() - startTime
        print(f'Planning time: {planning_time} seconds')
        average_reward = get_average_reward(a, env)

        vi_time.append(planning_time)
        vi_average_reward.append(average_reward)

    print('==================== RTDP ====================')
    for iteration in range(200):
        startTime = time.time()
        a = rtdpAgents.RTDPAgent(mdp, discount, iteration)

        planning_time = time.time() - startTime
        print(f'Planning time: {planning_time} seconds')
        average_reward = get_average_reward(a, env)

        rtdp_time.append(planning_time)
        rtdp_average_reward.append(average_reward)

    print("")
    print("### Plot Average Reward vs Time.")
    fig, ax = plt.subplots()
    ax.grid(True)

    plt.plot(vi_time, vi_average_reward, label = 'Value Iteration')
    plt.plot(rtdp_time, rtdp_average_reward, label = 'RTDP')
    plt.xlabel('Time')
    plt.ylabel('Average Reward')
    plt.title('Compare Algorithm')
    plt.legend()

    print("Close the plot diagram to continue program")
    plt.show()
