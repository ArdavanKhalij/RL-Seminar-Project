import numpy as np
import torch
import copy

def get_parameters(policy):
    parameters_tensor = list(policy.parameters())
    return parameters_tensor

def produce_perturbation(parameters):
    perturbation = []
    for params in parameters:
        perturb = torch.randn_like(params)
        perturbation.append(perturb)
    return perturbation

def add_list(x1, x2):
    result = []
    for (i, j) in zip(x1, x2):
        x = torch.tensor(i + j, requires_grad=False)
        result.append(x)
    return result

def diff_list(x1, x2):
    result = []
    for (i, j) in zip(x1, x2):
        x = torch.tensor(i - j, requires_grad=False)
        result.append(x)
    return result

def set_parameters(new_parameters, policy):
    for (name, param), new_param in zip(policy.named_parameters(), new_parameters):
        param.data = new_param
    return policy

def Zeroth_order(policy, env, learning_rate, perturbation_scale, num_episodes, num_iterations):
    results = []
    Best_score = float("-inf")
    for i in range(num_iterations):
        # Step 1: Produce perturbation vector
        parameters = get_parameters(policy)
        perturbation = produce_perturbation(parameters)

        # Step 2: Produce perturbations of θ
        theta_plus = add_list(parameters, perturbation)
        theta_minus = diff_list(parameters, perturbation)

        # Step 3: Evaluate perturbations
        score_plus = evaluate_policy(env, policy, theta_plus, num_episodes)
        score_minus = evaluate_policy(env, policy, theta_minus, num_episodes)

        # Step 4: Compute gradient
        gradient = 0.5 * (score_plus - score_minus) * np.array(perturbation)

        # Step 5: Update θ
        updated_params = add_list(parameters, learning_rate * gradient)
        policy = set_parameters(updated_params, policy)
        result = evaluate_policy(env, policy, updated_params, num_episodes)

        if Best_score < result:
            Best_score = result

        results.append(result)

        if i % 10 == 0:
            print(i, ".", result)

    return results



def evaluate_policy(env, policy, parameters, num_episodes):
    x_policy = copy.deepcopy(policy)
    x_policy = set_parameters(parameters, x_policy)
    episode_rewards = []
    for _ in range(num_episodes):
        state = torch.tensor(env.reset()[0], dtype=torch.float32).unsqueeze(0)
        state = state[0]
        done = False
        episode_reward = 0
        counter = 0
        while not done:
            counter = counter + 1
            action = x_policy(state).tolist()
            next_step = env.step(action)
            next_state = torch.tensor(next_step[0])
            reward = next_step[1]
            done = next_step[2]

            episode_reward += reward
            state = next_state
            if counter == 1000:
                break

        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards)
