import gym

import random
from collections import defaultdict
import numpy as np

import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MaxNLocator
from functools import partial
%matplotlib inline

matplotlib.style.use('ggplot')

def get_action(Q, state, epsilon):
    random_action = random.randint(0, 1)
    best_action = np.argmax(Q[state])
    return np.random.choice([best_action, random_action], p=[1 - epsilon, epsilon]) 

def evaluate_policy(Q, evaluating_episodes):
    wins = 0
    for _ in range(evaluating_episodes):
        state = env.reset()[0]
        
        done = False
        while not done:
            action = np.argmax(Q[state])
            
            state, reward, done, _ = env.step(action=action)[:-1]
            
        if reward > 0:
            wins += 1
        
    return wins / evaluating_episodes

def get_glie_epsilon(episodes):
    # return 1/episodes
    if episodes == 0:
        epsilon = 0
    else:
        eps_start=1.0 
        eps_decay=0.99999
        eps_min=0

        epsilon = eps_start*(eps_decay**(episodes-1))

    return epsilon

# constant alpha
def glie_first_visit_mc_control(env, episodes, gamma, evaluating_episodes):

    Q = defaultdict(lambda: np.zeros(env.action_space.n)) # value function
    policy = defaultdict(lambda: np.zeros(env.action_space.n)) 

    state_count = defaultdict(float) # state
    state_action_count = defaultdict(float) # increment counter
    
    evaluations = [] # keeping track of our policy evaluations
    evaluate = False
    
    for i in range(episodes):
        # evaluating a policy every 1000 games
        if evaluate and i % 1000 == 0:
            evaluations.append(evaluate_policy(Q, evaluating_episodes))
            print(f'at {i}')
    
        episode = [] # record states and actions to update value function
        
        # Game starting
        state = env.reset()[0]
        done = False
        
        # generating episodes
        while not done:
            state_count[state] += 1
            action = get_action(Q, state, get_glie_epsilon(i+1)) if state in Q else env.action_space.sample()
            policy[state] = action
            
            new_state, reward, done, _ = env.step(action=action)[:-1]
            
            episode.append((state, action, reward))
            state = new_state

        # at this point the game is finished, we either won or lost
        G = 0
        visited_state = [] # first visit count
        
        for s, a, r in reversed(episode):
            if episode.count((s, a, r)) == 1:
                if (s, a, r) not in visited_state:
                    visited_state.append((s, a, r))
                    
                    new_s_a_count = state_action_count[(s, a)] + 1
                    G = r + gamma * G
                    state_action_count[(s, a)] = new_s_a_count
                    Q[s][a] = Q[s][a] + (G - Q[s][a]) / new_s_a_count
        
        evaluate = True
        # print(Q[s])
            
    return Q, evaluations, policy

def plot_value_function(q, ax1, ax2):
    plt.figure(figsize=(25,20))
    player_sum = np.arange(12, 21 + 1)
    dealer_show = np.arange(1, 10 + 1)
    # player_sum = np.arange(1, 10 + 1)
    # dealer_show = np.arange(12, 21 + 1)
    usable_ace = np.array([False, True])
    state_values = np.zeros((len(player_sum), len(dealer_show), len(usable_ace)))

    for i, player in enumerate(player_sum):
        for j, dealer in enumerate(dealer_show):
            for k, ace in enumerate(usable_ace):
                state_values[i, j, k] = max(q[player, dealer, ace])
    
    X, Y = np.meshgrid(dealer_show, player_sum)
 
    ax1.plot_wireframe(X, Y, state_values[:, :, 0])
    ax2.plot_wireframe(X, Y, state_values[:, :, 1])
 
    for ax in ax1, ax2:
        ax.set_zlim(-1, 1)
        ax.set_ylabel('player sum')
        ax.set_xlabel('dealer showing')
        ax.set_zlabel('state-value')
        
def get_Z(player_hand, dealer_showing, usable_ace, policy):
    if (player_hand, dealer_showing, usable_ace) in policy:
        return policy[player_hand, dealer_showing, usable_ace]
    else:
        return 1

def get_figure(usable_ace, ax, policy):
    x_range = np.arange(1, 11)
    y_range = np.arange(11, 22)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([[get_Z(player_hand, dealer_showing, usable_ace, policy) for dealer_showing in x_range] for player_hand in range(21, 10, -1)])
    surf = ax.imshow(Z, cmap=plt.get_cmap('Accent', 2), vmin=0, vmax=1, extent=[0.5, 10.5, 10.5, 21.5])
    plt.xticks(x_range, ('A', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
    plt.yticks(y_range)
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Player Sum')
    ax.grid(color='black', linestyle='-', linewidth=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(surf, ticks=[0, 1], cax=cax)
    cbar.ax.set_yticklabels(['0 (STICK)','1 (HIT)'])
    cbar.ax.invert_yaxis()     
    
def plot_policy(policy):           
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(121)
    ax.set_title('Usable Ace', fontsize=16)
    get_figure(True, ax, policy)
    ax = fig.add_subplot(122)
    ax.set_title('No Usable Ace', fontsize=16)
    get_figure(False, ax, policy)
    plt.show()
    
env = gym.make('Blackjack-v1')

q, evaluations, policy = glie_first_visit_mc_control(env, 1000000, 0.4, 10000)

fig, axes = pyplot.subplots(nrows=2, figsize=(20, 15), subplot_kw={'projection': '3d'})
axes[0].set_title('value function without usable ace')
axes[1].set_title('value function with usable ace')
plot_value_function(q, axes[0], axes[1])

plt.figure(figsize=(10,5))
plt.plot([i * 1000 for i in range(len(evaluations))], evaluations)
plt.title('Change of win rate')
plt.xlabel('episode')
plt.ylabel('win rate')

p = dict((s,np.max(v)) for s, v in policy.items())
plot_policy(p)