from rl_mdp.mdp.reward_function import RewardFunction
from rl_mdp.mdp.transition_function import TransitionFunction
from rl_mdp.mdp.mdp import MDP
from rl_mdp.model_free_prediction.monte_carlo_evaluator import MCEvaluator
from rl_mdp.policy.policy import Policy
from rl_mdp.model_free_prediction.td_evaluator import TDEvaluator
from rl_mdp.model_free_prediction.td_lambda_evaluator import TDLambdaEvaluator
from rl_mdp.util import create_mdp, create_policy_1, create_policy_2

def compare_polcies(policy1_eval, policy2_eval, Algorithm):
    differences = policy1_eval - policy2_eval
    count = 0

    for diff in differences:
        count += diff
            
    if count < 0:
        result = ("Policy 2 is better than Policy 1 (π2 > π1) for\n"
                  f"the {Algorithm} algorithm.")
    elif count > 0:
        result = ("Policy 1 is better than Policy 2 (π1 > π2) for\n"
                  f"the {Algorithm} algorithm.")
    else:
        result = ("It cannot be concluded which policy is better\n"
                  "as the inequality v_π1(s)>=v_π2(s) does not hold for every s in S!")
    
    return result

def simple_print(Evaluation, policy_name, Evaluator):
    upper = f"[---------------------------{Evaluator}---------------------------]"
    lower = "-" * (len(upper)-2)
    print(upper)
    print(f"The estimates for the state value function for {policy_name} using {Evaluator} are: ")
    for count, i in enumerate(Evaluation):
        print(f"V(s{count}) = {i}")
    print(f"[{lower}]")
    
def main() -> None:
    """
    Starting point of the program, you can instantiate any classes, run methods/functions here as needed.
    """
    mdp = create_mdp()

    policy1 = create_policy_1()
    policy2 = create_policy_2()

    Evaluator_p1 = MCEvaluator(mdp)\
        .evaluate(policy1, 1000)
    
    Evaluator_p2 = MCEvaluator(mdp)\
        .evaluate(policy2, 1000)
    
    MC_comparison = compare_polcies(Evaluator_p1, Evaluator_p2, "MC first-visit")
    
    simple_print(Evaluator_p1, "Policy 1", "MC first-visit")
    simple_print(Evaluator_p2, "Policy 2", "MC first-visit")
    print(MC_comparison)

    TD_evaluator_p1 = TDEvaluator(mdp)\
        .evaluate(policy1, 1000)
    
    TD_evaluator_p2 = TDEvaluator(mdp)\
        .evaluate(policy2, 1000)
    
    TD_comparison = compare_polcies(TD_evaluator_p1, TD_evaluator_p2, "TD(0)")
    
    simple_print(TD_evaluator_p1, "Policy 1", "TD(0)")
    simple_print(TD_evaluator_p2, "Policy 2", "TD(0)")
    print(TD_comparison)

    TD_lambda_p1 = TDLambdaEvaluator(mdp)\
        .evaluate(policy1, 1000)
    
    TD_lambda_p2 = TDLambdaEvaluator(mdp)\
        .evaluate(policy2, 1000)
    
    TD_lambda_comparison = compare_polcies(TD_lambda_p1, TD_lambda_p2, "TD(lambda)")
    
    simple_print(TD_lambda_p1, "Policy 1", "TD(lambda)")
    simple_print(TD_lambda_p2, "Policy 2", "TD(lambda)")
    print(TD_lambda_comparison)

if __name__ == "__main__":
    main()
