from rl_mdp.mdp.reward_function import RewardFunction
from rl_mdp.mdp.transition_function import TransitionFunction
from rl_mdp.mdp.mdp import MDP
from rl_mdp.model_free_prediction.monte_carlo_evaluator import MCEvaluator
from rl_mdp.policy.policy import Policy
from rl_mdp.model_free_prediction.td_evaluator import TDEvaluator

import numpy as np

def creating_the_mdp():
    """
    Function that creates the mdp
    """
    states = [0, 1, 2, 3]    # Set of states actions represented as a list of integers.
    actions = [0, 1]

    # Define rewards using a dictionary
    rewards = {
        (0, 0): 0.0,           # state 0, action 0 gets reward -1.
        (0, 1): 0.0,
        (1, 0): 5.0,
        (1, 1): -1.0,
        (2, 0): -1.0,
        (2, 1): 10.0
    }

    # Create the RewardFunction object
    reward_function = RewardFunction(rewards)

    # Define transition probabilities using a dictionary
    transitions = {
        (0, 0): np.array([0, 0, 1, 0]),      # For state one, action one we get probability vector (0.7, 0.2, 0.1) representing the probability to transition to state 0, 1, 2 respectively.
        (0, 1): np.array([0, 0.8, 0.2, 0]),
        (1, 0): np.array([0, 0, 0.5, 0.5]),
        (1, 1): np.array([0, 1, 0, 0]),
        (2, 0): np.array([0, 0, 0, 1]),
        (2, 1): np.array([0, 0, 1, 0])
    }

    # Create the TransitionFunction object
    transition_function = TransitionFunction(transitions)

    # Create the MDP object
    mdp = MDP(states, actions, transition_function, reward_function, discount_factor=0.9, terminal_state = 3)
    
    return mdp


def policy_1():
    """
    Setting policy 1
    """
    policy = Policy()
    policy.set_action_probabilities(0, [0.7, 0.3])
    policy.set_action_probabilities(1, [0.6, 0.4])
    policy.set_action_probabilities(2, [0.9, 0.1])
    return policy

def policy_2():
    """
    Setting policy 2
    """
    policy = Policy()
    policy.set_action_probabilities(0, [0.5, 0.5])
    policy.set_action_probabilities(1, [0.3, 0.7])
    policy.set_action_probabilities(2, [0.8, 0.2])
    return policy

def simple_print(Evaluation, policy_name, Evaluator):
    print(f"The estimates for the state value function for {policy_name} using {Evaluator} are: ")
    for count, i in enumerate(Evaluation):
        print(f"V(s{count}) = {i}")
    

def main() -> None:
    """
    Starting point of the program, you can instantiate any classes, run methods/functions here as needed.
    """
    mdp = creating_the_mdp()

    policy1 = policy_1()
    policy2 = policy_2()

    Evaluator_p1 = MCEvaluator(mdp)\
        .evaluate(policy1, 1000)
    
    Evaluator_p2 = MCEvaluator(mdp)\
        .evaluate(policy2, 1000)
    
    simple_print(Evaluator_p1, "Policy 1", "MC first-visit")
    simple_print(Evaluator_p2, "Policy 2", "MC first-visit")

    TD_evaluator_p1 = TDEvaluator(mdp)\
        .evaluate(policy1, 1000)
    
    TD_evaluator_p2 = TDEvaluator(mdp)\
        .evaluate(policy2, 1000)
    
    simple_print(TD_evaluator_p1, "Policy 1", "TD(0)")
    simple_print(TD_evaluator_p2, "Policy 2", "TD(0)")

if __name__ == "__main__":
    main()
