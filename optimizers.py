"""
Optimizer prompt templates used in the paper.
See https://arxiv.org/abs/2406.04344 Appendix F for more details.
"""

def generate_opt_prompt_v1(theta, opt_def, one_batch_x, one_batch_y_hat, one_batch_y, verbose=True):
    opt =   f"## Optimization Step ##\n\n" + \
            f"You are the optimizer for a model, your goal is to learn the best descriptions for the model. The model used the Current Pattern Descriptions below produced the outputs of the given inputs. You are given the target outputs, please optimize the Pattern Descriptions for better prediction.\n\n" + \
            f"** Inputs (a batch of i.i.d. data): **\n{one_batch_x}\n\n" + \
            f"** Current Pattern Descriptions: **\n{theta}\n\n" + \
            f"** The model outputs: **\n{one_batch_y_hat}\n\n" + \
            f"** The target outputs: **\n{one_batch_y}\n\n" + \
            f"{opt_def} Please think step by step and give your outputs strictly in the following format:\n\n" + \
            f"```\n" + \
            f"Reasoning: \n[be explicit and verbose, improve the Current Pattern Descriptions by yourself; please show your work; note that you don't have access to computer]\n\n" + \
            f"New Pattern Descriptions: \n[put your new descriptions here; MUST be specific and concrete; ****MUST provide the exact value of the parameters if the descriptions potentially involve unknown or learnable parameters!!!****]\n```" + \
            f"\nPlease ONLY reply according to this format, don't give me any other words."
    
    if verbose:
        print(f"************ Optimization Prompt ************")
        print(opt)
        
    return opt


generate_opt_prompts = {
    "regression": generate_opt_prompt_v1,
}