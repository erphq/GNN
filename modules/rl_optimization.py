#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reinforcement Learning module for process optimization
"""

import numpy as np
import random

class ProcessEnv:
    """
    Environment for process optimization using RL
    The agent chooses (next_activity, resource) pairs and receives rewards
    based on cost, delay, and resource utilization
    """
    def __init__(self, df, le_task, resources):
        self.df = df
        self.le_task = le_task
        self.all_tasks = sorted(df["task_id"].unique())
        self.resources = resources
        self.start_task_id = 0
        self.done = False
        self.current_task = None
        
        # Additional state information could be added here
        self.resource_usage = {r: 0 for r in resources}
        self.total_cost = 0
        self.total_delay = 0
        
    def reset(self):
        """Reset the environment to initial state"""
        self.current_task = self.start_task_id
        self.done = False
        self.resource_usage = {r: 0 for r in self.resources}
        self.total_cost = 0
        self.total_delay = 0
        return self._get_state()
    
    def _get_state(self):
        """
        Get current state representation
        Currently using one-hot encoding for current task
        Could be extended with more features
        """
        # Size by the full label space, not `all_tasks`. Some encoded
        # task_ids only appear as `next_task` and are dropped from `df`
        # before `all_tasks` is computed, but `current_task` can still
        # take any value in [0, len(le_task.classes_)).
        state_vec = np.zeros(len(self.le_task.classes_), dtype=np.float32)
        state_vec[self.current_task] = 1.0
        return state_vec
    
    def step(self, action):
        """
        Take a step in the environment
        action = (next_activity_id, resource_id)
        Returns: (next_state, reward, done, info)
        """
        next_task, resource = action
        
        if next_task not in self.all_tasks:
            # Invalid action
            reward = -100.0
            self.done = True
            return self._get_state(), reward, self.done, {}
        
        # Compute costs and delays
        transition_cost = self._compute_transition_cost(self.current_task, next_task)
        processing_delay = self._compute_processing_delay(next_task, resource)
        resource_efficiency = self._compute_resource_efficiency(resource)
        
        # Update internal state
        self.total_cost += transition_cost
        self.total_delay += processing_delay
        self.resource_usage[resource] += 1
        
        # Compute reward components
        cost_penalty = -transition_cost
        delay_penalty = -processing_delay
        efficiency_bonus = resource_efficiency
        
        # Combined reward
        reward = cost_penalty + delay_penalty + efficiency_bonus
        
        # Move to next state
        self.current_task = next_task
        
        # Check if process should end
        if self._should_terminate():
            self.done = True
        
        info = {
            'transition_cost': transition_cost,
            'processing_delay': processing_delay,
            'resource_efficiency': resource_efficiency
        }
        
        return self._get_state(), reward, self.done, info
    
    def _compute_transition_cost(self, current_task, next_task):
        """
        Compute cost of transitioning between tasks
        Currently using a simple distance metric
        Could be replaced with actual cost data
        """
        return abs(next_task - current_task) * 1.0
    
    def _compute_processing_delay(self, task, resource):
        """
        Compute processing delay for task-resource pair
        Currently using random delays
        Could be replaced with historical data
        """
        base_delay = random.random() * 2.0
        resource_factor = 1.0 + (self.resource_usage[resource] * 0.1)
        return base_delay * resource_factor
    
    def _compute_resource_efficiency(self, resource):
        """
        Compute resource utilization efficiency
        Rewards balanced resource usage
        """
        total_usage = sum(self.resource_usage.values())
        if total_usage == 0:
            return 1.0
        
        current_usage = self.resource_usage[resource]
        expected_usage = total_usage / len(self.resources)
        
        if current_usage <= expected_usage:
            return 1.0
        else:
            return max(0.0, 1.0 - (current_usage - expected_usage) * 0.1)
    
    def _should_terminate(self):
        """
        Determine if the process should terminate
        Currently using a simple random termination
        Could be replaced with actual process end conditions
        """
        return random.random() < 0.1

def run_q_learning(env, episodes=30, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Q-learning algorithm for process optimization
    
    Parameters:
    - env: ProcessEnv instance
    - episodes: Number of training episodes
    - alpha: Learning rate
    - gamma: Discount factor
    - epsilon: Exploration rate
    
    Returns:
    - Q-table mapping state-action pairs to values
    """
    possible_tasks = env.all_tasks
    possible_resources = env.resources
    
    # All possible actions (task, resource pairs)
    all_actions = []
    for t in possible_tasks:
        for r in possible_resources:
            all_actions.append((t, r))
    num_actions = len(all_actions)
    
    Q_table = {}
    
    def get_state_key(state):
        """Convert state array to hashable tuple"""
        return tuple(state.round(3))
    
    def get_Q(state):
        """Get Q-values for state, initialize if needed"""
        sk = get_state_key(state)
        if sk not in Q_table:
            Q_table[sk] = np.zeros(num_actions, dtype=np.float32)
        return Q_table[sk]
    
    # Training loop
    for ep in range(episodes):
        s = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # ε-greedy action selection
            if random.random() < epsilon:
                action_idx = random.randrange(num_actions)
            else:
                q_values = get_Q(s)
                action_idx = int(np.argmax(q_values))
            
            action = all_actions[action_idx]
            next_state, reward, done, _info = env.step(action)
            total_reward += reward
            
            # Q-learning update
            current_q = get_Q(s)
            next_q = get_Q(next_state)
            best_next_q = 0.0 if done else np.max(next_q)
            
            # Update Q-value
            current_q[action_idx] += alpha * (
                reward + gamma * best_next_q - current_q[action_idx]
            )
            
            s = next_state
        
        print(f"Episode {ep+1}/{episodes}, total_reward={total_reward:.2f}")
    
    return Q_table

def get_optimal_policy(Q_table, all_actions):
    """
    Extract optimal policy from Q-table
    
    Returns:
    - Dictionary mapping states to optimal actions
    """
    policy = {}
    for state in Q_table:
        q_values = Q_table[state]
        optimal_action_idx = np.argmax(q_values)
        policy[state] = all_actions[optimal_action_idx]
    return policy 