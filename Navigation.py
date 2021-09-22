'''
[Deep Reinforcement Learning Nanodegree]
(https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).
'''
import os

from unityagents import UnityEnvironment
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from dqn_agent import Agent


def train(env, agent, n_episodes=1800, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
  """Deep Q-Learning.

  Params
  ======
      env (UnityEnvironment): UnityEnvironment
      n_episodes (int): maximum number of training episodes
      max_t (int): maximum number of timesteps per episode
      eps_start (float): starting value of epsilon, for epsilon-greedy action selection
      eps_end (float): minimum value of epsilon
      eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
  """
  brain_name = env.brain_names[0]
  brain = env.brains[brain_name]

  scores = []                        # list containing scores from each episode
  scores_window = deque(maxlen=100)  # last 100 scores
  eps = eps_start                    # initialize epsilon
  for i_episode in range(1, n_episodes+1):
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    for t in range(max_t):
      action = agent.act(state, eps).astype(int)
      env_info = env.step(action)[brain_name]
      next_state, reward, done = env_info.vector_observations[
          0], env_info.rewards[0], env_info.local_done[0]
      agent.step(state, action, reward, next_state, done)
      state = next_state
      score += reward
      if done:
        break
    scores_window.append(score)       # save most recent score
    scores.append(score)              # save most recent score
    eps = max(eps_end, eps_decay*eps)  # decrease epsilon
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(
        i_episode, np.mean(scores_window)), end="")
    if i_episode % 100 == 0:
      print('\rEpisode {}\tAverage Score: {:.2f}'.format(
          i_episode, np.mean(scores_window)))
      torch.save(agent.qnetwork_local.state_dict(),
                 os.getcwd() + f'/checkpoints/checkpoint_dqn_{i_episode}.pth')
    if np.mean(scores_window) >= 20.0:
      print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
          i_episode-100, np.mean(scores_window)))
      torch.save(agent.qnetwork_local.state_dict(),
                 os.getcwd() + f'/checkpoints/checkpoint_dqn_last.pth')
      break
  return scores


def test(env, agent, n_episodes=100, max_t=1000, saved_model='checkpoint.pth'):
  """Deep Q-Learning.

  Params
  ======
      env : UnityEnvironment
      agent : algoritm
      n_episodes (int): maximum number of training episodes
      max_t (int): maximum number of timesteps per episode
  """
  brain_name = env.brain_names[0]
  brain = env.brains[brain_name]

  agent.qnetwork_local.load_state_dict(torch.load(saved_model))
  print('='*30)
  print('Sucessfully loaded from {}'.format(saved_model))
  print('='*30)

  scores = []                        # list containing scores from each episode
  # scores_window = deque(maxlen=100)  # last 100 scores
  # eps = eps_start                    # initialize epsilon
  for i_episode in range(1, n_episodes+1):
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    for t in range(max_t):
      action = agent.act(state, 0.).astype(int)
      env_info = env.step(action)[brain_name]
      next_state, reward, done = env_info.vector_observations[
          0], env_info.rewards[0], env_info.local_done[0]
      agent.step(state, action, reward, next_state, done)
      state = next_state
      score += reward
      if done:
        break
    scores.append(score)
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(
        i_episode, np.mean(scores)), end="")
    if i_episode % 100 == 0:
      print('\rEpisode {}\tAverage Score: {:.2f}'.format(
          i_episode, np.mean(scores)))

  print('='*30)
  print("Test Average Score: {:.2f}".format(np.mean(scores)))
  print('='*30)

  return scores


if __name__ == '__main__':
  # [mode select(train/test/both)]
  mode = 'both'

  env = UnityEnvironment(file_name=os.getcwd() +
                         "/Banana_Windows_x86_64/Banana.exe")

  agent = Agent(state_size=37, action_size=4, seed=2021)

  train_n_episodes = 1500
  if mode == 'train':
    scores_train = train(env, agent, n_episodes=train_n_episodes)
  elif mode == 'test':
    saved_model = os.getcwd() + f'/checkpoints/checkpoint_dqn_1500.pth'
    scores_test = test(env, agent, n_episodes=100, saved_model=saved_model)
  else:
    scores_train = train(env, agent, n_episodes=train_n_episodes)
    saved_model = os.getcwd() + \
        f'/checkpoints/checkpoint_dqn_{train_n_episodes}.pth'
    scores_test = test(env, agent, n_episodes=100, saved_model=saved_model)

  env.close()
