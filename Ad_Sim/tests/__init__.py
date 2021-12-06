'''
       # Plot CTRs and probabilities
       plt.subplot2grid(grid_size, (2, 1), rowspan=3, colspan=1)
       x = [ad.id for ad in ads]
       y = [ad.ctr() for ad in ads]
       y_2 = self.click_probabilities
       x_pos = [i for i, _ in enumerate(x)]
       x_pos_2 = [i + 0.4 for i, _ in enumerate(x)]
       plt.ylabel("Ads")
       plt.xlabel("")
       plt.yticks(x_pos, x)
       plt.barh(x_pos, y, 0.4, label='Actual CTR')
       plt.barh(x_pos_2, y_2, 0.4, label='Probability')
       plt.legend(loc='upper right')
       '''


'''
def evaluate_training_result(env, agent):
    total_reward = 0.0
    print("Hello let us begin")
    episodes_to_play = 1
    for i in range(episodes_to_play):
        state = env.reset(agent.scenario_name)
        done = False
        episode_reward = 0.0
        while not done:
            action = agent.policy(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_reward += episode_reward
    average_reward = total_reward / episodes_to_play
    return average_reward
'''
'''
       for state in state_batch:
          state_input = self.create_input(state)
          new_state_batch = np.append(new_state_batch, state_input)
       new_state_batch = np.reshape(new_state_batch, (1, -1))
       new_state_batch = tf.convert_to_tensor(new_state_batch, dtype=tf.float32)

       for next_state in next_state_batch:
           state_input = self.create_input(next_state)
           new_next_state_batch = np.append(new_next_state_batch, state_input)
       new_next_state_batch = np.reshape(new_next_state_batch, (1, -1))
       new_next_state_batch = tf.convert_to_tensor(new_next_state_batch, dtype=tf.float32)

       current_q = self.q_net(new_state_batch).numpy() #10 values
       target_q = np.copy(current_q)
       print(target_q)
       next_q = self.target_q_net(new_next_state_batch).numpy()
       max_next_q = np.amax(next_q, axis=1)
       for i in range(state_batch.shape[0]):
           target_q_val = reward_batch[i]
           if not done_batch[i]:
               target_q_val += 0.95 * max_next_q
           target_q[0][action_batch[i]] = target_q_val
       training_history = self.q_net.fit(x=new_state_batch, y=target_q, verbose=0)
       loss = training_history.history['loss']
       return loss
       '''