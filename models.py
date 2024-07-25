"""
Model prompt templates used in the paper.
See https://arxiv.org/abs/2406.04344 Appendix F for more details.
"""

def generate_forward_prompt_regression(theta, x, verbose=True):
    fx =    f"## Inference Step ##\n\n" + \
            f"You are the model. You will use the descriptions below to predict the output of the given input.\n\n" + \
            f"** Pattern Descriptions: **\n{theta}\n\n" + \
            f"** Input: **\n{x}\n\n" + \
            f"Please give your output strictly in the following format:\n\n" + \
            f"```\nExplanations: [Your step-by-step analyses and results]\n\n" + \
            f"Output: \n[Your output MUST be in REAL NUMBER ROUNDED TO TWO DECIMAL POINTS; make necessary assumptions if needed; it MUST be in the same format as the Input]\n```" + \
            f"\nPlease ONLY reply according to this format, don't give me any other words."

    if verbose:
        print(f"************ Forward Pass Prompt Example ************")
        print(fx)

    return fx


generate_forward_prompts = {
    "regression": generate_forward_prompt_regression
}