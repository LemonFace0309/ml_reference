# UNQ_C1
# GRADED CELL
# Create the Q-Network
q_network = Sequential([
    ### START CODE HERE ### 
    Input(shape=state_size),
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dense(num_actions, activation="linear")
    ### END CODE HERE ### 
    ])

# Create the target Q^-Network
target_q_network = Sequential([
    ### START CODE HERE ### 
    Input(shape=state_size),
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dense(num_actions, activation="linear")
    ### END CODE HERE ###
    ])

### START CODE HERE ### 
optimizer = Adam(learning_rate=ALPHA)
### END CODE HERE ###


# UNQ_C2
# GRADED FUNCTION: calculate_loss
# Computes loss between the y targets and the Q(s, a) values
def compute_loss(experiences, gamma, q_network, target_q_network):
    """ 
    Calculates the loss.
    
    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
      q_network: (tf.keras.Sequential) Keras model for predicting the q_values
      target_q_network: (tf.keras.Sequential) Karas model for predicting the targets
          
    Returns:
      loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
            the y targets and the Q(s,a) values.
    """
    
    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences
    
    # Compute max Q^(s,a)
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
    
    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    ### START CODE HERE ### 
    y_targets = rewards +  gamma*(1 - done_vals)*max_qsa
    ### END CODE HERE ###
    
    # Get the q_values
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))
        
    # Compute the loss
    ### START CODE HERE ### 
    loss = MSE(y_targets, q_values) 
    ### END CODE HERE ### 
    
    return loss



### Update the Network Weights
@tf.function
def agent_learn(experiences, gamma):
    """
    Updates the weights of the Q networks.
    
    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
    
    """
    
    # Calculate the loss
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)

    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, q_network.trainable_variables)
    
    # Update the weights of the q_network.
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    # update the weights of target q_network
    utils.update_target_network(q_network, target_q_network)


### Train the Agent

# 1.  Initialize memory buffer D with capacity N
# 2.  Initialize Q-Network with random weight w
# 3.  Initialize target Q^-Network with weights w- = w
# 4.  for episode i = 1 to M do
# 5.    Receive initial observation state S_1
# 6.    for t = 1 to T do
# 7.      Observe states S_t and choose action A_t using an epsilon-greedy policy
# 8.      Take action A_t in the environment, receive reward R_t and next state S_t+1
# 9.      Store experience tuple (S_t, A_t, R_t, S_t+1) in memory buffer D
# 10.     Every C steps perform a learning update:
# 11.     Sample random mini-batch of experience tuples (S_j, A_j, R_j, S_j+1) from D
# 12.     Set y_j = R_j if episode terminate at step j+1, otherwise set y_i = R_j + gamma * max a' Q^(s_j+1, a')
# 13.     Perform a gradient descent step on (y_j - Q(s_j, a_j; w))^2 with respect to the Q-Network weights w
# 14.     Update the weights of the Q^-Network using a soft update
# 15.   end
# 16. end


# Line 1: We initialize the memory_buffer with a capacity of  𝑁=  MEMORY_SIZE. Notice that we are using a deque as the data structure for our memory_buffer.
# Line 2: We skip this line since we already initialized the q_network in Exercise 1.
# Line 3: We initialize the target_q_network by setting its weights to be equal to those of the q_network.
# Line 4: We start the outer loop. Notice that we have set  𝑀=  num_episodes = 2000. This number is reasonable because the agent should be able to solve the Lunar Lander environment in less than 2000 episodes using this notebook's default parameters.
# Line 5: We use the .reset() method to reset the environment to the initial state and get the initial state.
# Line 6: We start the inner loop. Notice that we have set  𝑇=  max_num_timesteps = 1000. This means that the episode will automatically terminate if the episode hasn't terminated after 1000 time steps.
# Line 7: The agent observes the current state and chooses an action using an  𝜖 -greedy policy. Our agent starts out using a value of  𝜖=  epsilon = 1 which yields an  𝜖 -greedy policy that is equivalent to the equiprobable random policy. This means that at the beginning of our training, the agent is just going to take random actions regardless of the observed state. As training progresses we will decrease the value of  𝜖  slowly towards a minimum value using a given  𝜖 -decay rate. We want this minimum value to be close to zero because a value of  𝜖=0  will yield an  𝜖 -greedy policy that is equivalent to the greedy policy. This means that towards the end of training, the agent will lean towards selecting the action that it believes (based on its past experiences) will maximize  𝑄(𝑠,𝑎) . We will set the minimum  𝜖  value to be 0.01 and not exactly 0 because we always want to keep a little bit of exploration during training. If you want to know how this is implemented in code we encourage you to take a look at the utils.get_action function in the utils module.
# Line 8: We use the .step() method to take the given action in the environment and get the reward and the next_state.
# Line 9: We store the experience(state, action, reward, next_state, done) tuple in our memory_buffer. Notice that we also store the done variable so that we can keep track of when an episode terminates. This allowed us to set the  𝑦  targets in Exercise 2.
# Line 10: We check if the conditions are met to perform a learning update. We do this by using our custom utils.check_update_conditions function. This function checks if  𝐶=  NUM_STEPS_FOR_UPDATE = 4 time steps have occured and if our memory_buffer has enough experience tuples to fill a mini-batch. For example, if the mini-batch size is 64, then our memory_buffer should have at least 64 experience tuples in order to pass the latter condition. If the conditions are met, then the utils.check_update_conditions function will return a value of True, otherwise it will return a value of False.
# Lines 11 - 14: If the update variable is True then we perform a learning update. The learning update consists of sampling a random mini-batch of experience tuples from our memory_buffer, setting the  𝑦  targets, performing gradient descent, and updating the weights of the networks. We will use the agent_learn function we defined in Section 8 to perform the latter 3.
# Line 15: At the end of each iteration of the inner loop we set next_state as our new state so that the loop can start again from this new state. In addition, we check if the episode has reached a terminal state (i.e we check if done = True). If a terminal state has been reached, then we break out of the inner loop.
# Line 16: At the end of each iteration of the outer loop we update the value of  𝜖 , and check if the environment has been solved. We consider that the environment has been solved if the agent receives an average of 200 points in the last 100 episodes. If the environment has not been solved we continue the outer loop and start a new episode.
# Finally, we wanted to note that we have included some extra variables to keep track of the total number of points the agent received in each episode. This will help us determine if the agent has solved the environment and it will also allow us to see how our agent performed during training. We also use the time module to measure how long the training takes.

start = time.time()

num_episodes = 2000
max_num_timesteps = 1000

total_point_history = []

num_p_av = 100    # number of total points to use for averaging
epsilon = 1.0     # initial ε value for ε-greedy policy

# Create a memory buffer D with capacity N
memory_buffer = deque(maxlen=MEMORY_SIZE)

# Set the target network weights equal to the Q-Network weights
target_q_network.set_weights(q_network.get_weights())

for i in range(num_episodes):
    
    # Reset the environment to the initial state and get the initial state
    state = env.reset()
    total_points = 0
    
    for t in range(max_num_timesteps):
        
        # From the current state S choose an action A using an ε-greedy policy
        state_qn = np.expand_dims(state, axis=0)  # state needs to be the right shape for the q_network
        q_values = q_network(state_qn)
        action = utils.get_action(q_values, epsilon)
        
        # Take action A and receive reward R and the next state S'
        next_state, reward, done, _ = env.step(action)
        
        # Store experience tuple (S,A,R,S') in the memory buffer.
        # We store the done variable as well for convenience.
        memory_buffer.append(experience(state, action, reward, next_state, done))
        
        # Only update the network every NUM_STEPS_FOR_UPDATE time steps.
        update = utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)
        
        if update:
            # Sample random mini-batch of experience tuples (S,A,R,S') from D
            experiences = utils.get_experiences(memory_buffer)
            
            # Set the y targets, perform a gradient descent step,
            # and update the network weights.
            agent_learn(experiences, GAMMA)
        
        state = next_state.copy()
        total_points += reward
        
        if done:
            break
            
    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-num_p_av:])
    
    # Update the ε value
    epsilon = utils.get_new_eps(epsilon)

    print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

    if (i+1) % num_p_av == 0:
        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

    # We will consider that the environment is solved if we get an
    # average of 200 points in the last 100 episodes.
    if av_latest_points >= 200.0:
        print(f"\n\nEnvironment solved in {i+1} episodes!")
        q_network.save('lunar_lander_model.h5')
        break
        
tot_time = time.time() - start

print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")
# Episode 100 | Total point average of the last 100 episodes: -150.85
# Episode 200 | Total point average of the last 100 episodes: -106.11
# Episode 300 | Total point average of the last 100 episodes: -77.256
# Episode 400 | Total point average of the last 100 episodes: -25.01
# Episode 500 | Total point average of the last 100 episodes: 159.91
# Episode 534 | Total point average of the last 100 episodes: 201.37

# Environment solved in 534 episodes!

# Total Runtime: 693.66 s (11.56 min)








######## utils.py
import base64
import random
from itertools import zip_longest

import imageio
import IPython
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import tensorflow as tf
from statsmodels.iolib.table import SimpleTable

SEED = 0              # seed for pseudo-random number generator
MINIBATCH_SIZE = 64   # mini-batch size
TAU = 1e-3            # soft update parameter
E_DECAY = 0.995       # ε decay rate for ε-greedy policy
E_MIN = 0.01          # minimum ε value for ε-greedy policy

def get_experiences(memory_buffer):
    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
    states = tf.convert_to_tensor(np.array([e.state for e in experiences if e is not None]),dtype=tf.float32)
    actions = tf.convert_to_tensor(np.array([e.action for e in experiences if e is not None]), dtype=tf.float32)
    rewards = tf.convert_to_tensor(np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32)
    next_states = tf.convert_to_tensor(np.array([e.next_state for e in experiences if e is not None]),dtype=tf.float32)
    done_vals = tf.convert_to_tensor(np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
                                     dtype=tf.float32)
    return (states, actions, rewards, next_states, done_vals)

def check_update_conditions(t, num_steps_upd, memory_buffer):
    if (t + 1) % num_steps_upd == 0 and len(memory_buffer) > MINIBATCH_SIZE:
        return True
    else:
        return False

def get_new_eps(epsilon):
    return max(E_MIN, E_DECAY*epsilon)

def get_action(q_values, epsilon=0):
    if random.random() > epsilon:
        return np.argmax(q_values.numpy()[0])
    else:
        return random.choice(np.arange(4))
    
    
def update_target_network(q_network, target_q_network):
    for target_weights, q_net_weights in zip(target_q_network.weights, q_network.weights):
        target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)
