#######################################################################################################################
# Libraries
#######################################################################################################################
import torch
#######################################################################################################################



#######################################################################################################################
# 3. Implement the simple Population Method described above to learn a policy in LunarLanderContinuous.
#######################################################################################################################
def simple_population(N, env, policy, episode_count, generation):
    List_of_scores = []
    # Best score set -inf at the start, so we can detect all possible scores
    best_score = float("-inf")
    best_params = policy.state_dict()
    best_params_x = best_params
    policy_x = policy

    # Set of perturbed parameters
    perturbed_params = {}

    for i in range(0, generation):
        # Making new parameters
        REW = 0
        params_for_next_list = []
        scores_for_next_list = []
        for _ in range(0, N):
            # Generate random noise with the same shape as the parameter
            for name, param in policy_x.named_parameters():
                # Generate random noise with the same shape as the parameter
                noise = torch.randn_like(param)
                perturbed_params[name] = param + noise

            # Update the policy with perturbed parameters
            policy_x.load_state_dict(perturbed_params)

            # Evaluate the perturbed policy
            total_reward = 0
            for _ in range(episode_count):
                state = torch.tensor(env.reset()[0], dtype=torch.float32).unsqueeze(0)
                done = False
                counter = 0
                while not done:
                    counter = counter + 1
                    action = policy_x(state).tolist()[0]
                    next_step = env.step(action)
                    next_state = torch.tensor(next_step[0], dtype=torch.float32).unsqueeze(0)
                    reward = next_step[1]
                    done = next_step[2]
                    total_reward += reward
                    state = next_state
                    if counter == 1000:
                        break
            total_reward = total_reward / episode_count

            params_for_next_list.append(policy_x.state_dict())
            scores_for_next_list.append(total_reward)

            # Check if the perturbed policy has a better score
            if total_reward > best_score:
                best_score = total_reward
                best_params = policy_x.state_dict()

            REW = REW + total_reward

        List_of_scores.append(max(scores_for_next_list))
        if i%10 == 0:
            print(i, ". ", max(scores_for_next_list))

        max_index = scores_for_next_list.index(max(scores_for_next_list))

        # Update the policy with the best parameters
        policy_x.load_state_dict(params_for_next_list[max_index])

    return List_of_scores
#######################################################################################################################
