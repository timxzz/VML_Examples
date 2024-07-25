import os, time, re
import os.path as osp
import json
from openai import OpenAI, AsyncOpenAI
import wandb
import matplotlib.pyplot as plt
import asyncio
from argparse import ArgumentParser

import numpy as np
np.set_printoptions(precision=8, suppress=True)

import models
import optimizers

#  OpenAI
OPENAI_API_KEY = "<KEY>"

#  vLLM
VLLM_ENDPOINT = "http://<ENDPOINT_IP>:8000/v1"
VLLM_APIKEY = "token-abc123"

SYS_PROMPT = "You are a helpful, respectful and honest assistant."

def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


def create_batch(x, y=None, batch_size=10):
    x_batch = []
    if y is not None:
        y_batch = []
    for i in range(0, len(x), batch_size):
        x_batch.append(x[i:i+batch_size])
        if y is not None:
            y_batch.append(y[i:i+batch_size])

    if y is not None:
        x_batch
        return x_batch, y_batch
    else:
        return x_batch

def setup_api_client(api_key, base_url=None):
    if base_url is None:
        client = OpenAI(
            api_key=api_key,
        )

        async_client = AsyncOpenAI(
            api_key=api_key,
        )
    else:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    return client, async_client

######################## Training Utils ############################
def batch_generate_forward_prompt(fx_name, theta, one_batch_x, verbose=True):
    fx_prompt_batch = []
    i = 0
    for x in one_batch_x:
        if i > 0:
            verbose = False
        fx = models.generate_forward_prompts[fx_name](theta, x, verbose=verbose)
        fx_prompt_batch.append(fx)
        i += 1
    return fx_prompt_batch


async def execute_forward_pass(fx_prompt, llm_endpoint, llm_name, id, verbose=True):
    forward_state=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": fx_prompt}
            ]
    req_success = False
    while not req_success:
        try:
            completion = await llm_endpoint.chat.completions.create(
                    model=llm_name,
                    messages=forward_state
                )
            req_success = True
        except Exception as e:
            print(f"*** API Error *** ")
            print(e)
            print(f"Retrying... ID: {id}")

    forward_state.append({"role": "assistant", "content": completion.choices[0].message.content})

    if verbose:
        print(f"************ Forward Pass Output ************")
        print(completion.choices[0].message.content)

    return completion.choices[0].message.content, forward_state, id


async def batch_execute_forward_pass(fx_prompt_batch, id_batch, llm_endpoint, llm_name, verbose=True):
    tasks = [execute_forward_pass(fx_prompt, llm_endpoint, llm_name, id, verbose=verbose) for fx_prompt, id in zip(fx_prompt_batch, id_batch)]
    results = await asyncio.gather(*tasks)
    return results


def batch_inference(one_batch_X, llm_endpoint, llm_name, fx_name, theta, batch_size=None, verbose=False):
    if verbose:
        print()
        print()
        print(f"************ Batch Inference ************")
        print(f"************ *************** ************")

    if batch_size is None:
        batch_size = len(one_batch_X)
    y_hats = [None] * len(one_batch_X)
    done = False
    while not done:
        #  Get a batch of x to test
        x_batch = []
        x_batch_ids = []
        for i, y_hat in enumerate(y_hats):
            if y_hat is None:
                x_batch.append(one_batch_X[i])
                x_batch_ids.append(i)
            if len(x_batch) == batch_size:
                break
        #  If there are no more x to test, we are done
        if len(x_batch) == 0:
            done = True
            break

        #  Generate forward prompt for the batch
        fx_prompt_batch = batch_generate_forward_prompt(fx_name, theta, x_batch, verbose=verbose)
        #  Execute forward pass for the batch
        result_lists = asyncio.run(batch_execute_forward_pass(fx_prompt_batch, x_batch_ids, llm_endpoint, llm_name, verbose=verbose))
        for (forward_output, forward_state, id) in result_lists:
            y_hat = extract_y_hat(forward_output)


            if y_hat is not None and y_hat.shape == one_batch_X[id].shape:
                y_hats[id] = y_hat
            elif y_hat is not None:
                print(y_hat.shape, one_batch_X[id])
                print(f"Output shape mismatch. Retrying... ID: {id}")
            else:
                print("!!!!!!!!!!!!!!!!!!!!")
                print(forward_output)
                print(f"Output is None. Retrying... ID: {id}")
        time.sleep(3.0)

    if verbose:
        print(f"************ *************** ************")
        print(f"************ *************** ************")
        print()
        print()

    return np.array(y_hats)


def execute_optimization(opt_prompt, llm_endpoint, llm_name, last_opt_state=None, verbose=True):
    if last_opt_state is not None:
        if len(last_opt_state) > 3: # length for: Order 1 Markov: 3, Order 2 Markov: 5, Order 3 Markov: 7
            opt_state = []
            opt_state.append(last_opt_state[0])
            opt_state.extend(last_opt_state[-2:])
            if verbose:
                print(f"************ Shortening the state from {len(last_opt_state)} to {len(opt_state)} ************")
        else:
            opt_state = last_opt_state
    else:
        opt_state=[
                {"role": "system", "content": SYS_PROMPT},
            ]
        
    opt_state.append({"role": "user", "content": opt_prompt})
    # if verbose:
    #     print("#######################################################################")
    #     print(opt_state)
    #     print("#######################################################################")
    completion = llm_endpoint.chat.completions.create(
            model=llm_name,
            messages=opt_state
        )
    
    opt_state.append({"role": "assistant", "content": completion.choices[0].message.content})
    
    if verbose:
        print(f"************ Optimization Output ************")
        print(completion.choices[0].message.content)

    return completion.choices[0].message.content, opt_state


def calculate_mse_loss(y_true, y_pred, verbose=True):
    l = np.mean((y_true - y_pred)**2)
    if verbose:
        print(f"************ Loss ************")
        print(l)
    return l

def calculate_mse_in_out_loss(X_plot, y_plot_hat, y_plot, x_left, x_right, verbose=True):
    in_idx = np.where((X_plot >= x_left) & (X_plot <= x_right))[0]
    out_idx = np.where((X_plot < x_left) | (X_plot > x_right))[0]
    in_loss = np.mean((y_plot_hat[in_idx] - y_plot[in_idx])**2)
    out_loss = np.mean((y_plot_hat[out_idx] - y_plot[out_idx])**2)
    overall_loss = np.mean((y_plot_hat - y_plot)**2)
    if verbose:
        print(f"************ In Loss ************")
        print(in_loss)
        print(f"************ Out Loss ************")
        print(out_loss)
        print(f"************ Overall Loss ************")
        print(overall_loss)
    return in_loss, out_loss, overall_loss


######################## Train & Test ############################
def train_step(theta,
                train_X_batches,
                train_y_batches,
                llm_endpoint,
                async_llm_endpoint,
                hp,
                epoch,
                i,
                last_opt_state=None,
                verbose=True):
    if verbose:   
        print(f"************ Epoch {epoch} - Step {i}  ************")

    # Point inference with batch optimization
    y_hats = batch_inference(train_X_batches[i], async_llm_endpoint, hp['llm_name'], hp['fx_name'], theta, batch_size=hp['batch_size'], verbose=verbose)
    loss = calculate_mse_loss(train_y_batches[i], y_hats, verbose=verbose)
    opt_prompt = optimizers.generate_opt_prompts[hp['opt_name']](theta, hp['opt_hp'], train_X_batches[i], y_hats, train_y_batches[i], verbose=verbose)
    opt_output, opt_state = execute_optimization(opt_prompt, llm_endpoint, hp['llm_name'], 
                                                last_opt_state=last_opt_state,
                                                verbose=verbose)
    theta = extract_theta(opt_output, verbose=verbose)
    assert theta is not None, "Theta is None"

    wandb.log({"train_loss": loss, "epoch": epoch, "step": i})

    return theta, loss, opt_state


def test(test_X, llm_endpoint, llm_name, fx_name, theta, batch_size, verbose=False):
    y_hats = batch_inference(test_X, llm_endpoint, llm_name, fx_name, theta, batch_size=batch_size, verbose=verbose)
    return y_hats
    

######################## Output Extract ############################
def extract_y_hat(output_text):
    """
    Output Text example:
    ```
    Explanations: 
    This is a regression model, but there is no training data provided, so I will assume a simple linear regression model with a slope of 1 and an intercept of 0. 
    The input is [0.88888889], so I will multiply it by the slope and add the intercept to get the output.

    Output: 
    [0.88888889]
    ```
    return a np.array of the predicted output
    """
    # Regular expression to locate the "Outputs:" section followed by a number
    section_pattern = re.compile(
        r"Output:\s*\n?\[?\s*(-?\d+\.?\d*)\s*\]?"
    )

    # Find the block of text for outputs
    section_match = section_pattern.search(output_text)
    if section_match:
        match = section_match.group(1)
        # Convert match to a numpy array
        y_hat = np.array([match], dtype=float)
        # round for numerical stability
        y_hat = np.round(y_hat, 2)
    else:
        print("No output section found.")
        y_hat = None
    return y_hat


def extract_theta(output_text, verbose=True):
    """
    Output Text example:
    ```
    Explanations: I will update my model by training a linear regression model on the given data. I will use the input data as the feature and the ground truth outputs as the target variable.

    Pattern Descriptions: I am a simple linear regression model that maps input real numbers to output real numbers using a learned linear function, where the coefficients are estimated by minimizing the mean squared error between the predicted and actual outputs.
    ```
    return text after 'Pattern Descriptions: '
    """
    # Regular expression to locate the "Pattern Descriptions:" section followed by the model description
    section_pattern = re.compile(r"\**New Pattern Descriptions:\**\s*\n?([\S\s]*)")
    section_match = section_pattern.search(output_text)
    if section_match:
        theta = section_match.group(1)
    else:
        print("No model specification found.")
        theta = None

    if verbose:
        print(f"************ Updated Theta ************")
        print(theta)
    return theta

######################## Plotting ############################


def plot_regression(train_X, train_y, plot_X, plot_y_hat, step, path=None):
    fig, ax = plt.subplots()
    ax.scatter(train_X, train_y, label='Training Data')
    ax.plot(plot_X, plot_y_hat, label='Regression Line', color='red')
    ax.set_title(f"Step {step}")
    ax.legend()
    if path is not None:
        plt.savefig(path)
    wandb.log({f"plots_step/linspace": wandb.Image(fig)})
    plt.close(fig)


# MARK: - Main START
######################## ############### ############################
######################## ############### ############################
######################## ############### ############################

parser = ArgumentParser()

parser.add_argument("--job_id", type=str, default="local")
parser.add_argument("--model", choices=["llama", "gpt-4o"], default="llama")
parser.add_argument("--task", choices=["linear_regression", 
                                       "poly_regression", 
                                       "sine_regression"])

parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--eval_batch_size", type=int, default=100)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--add_prior", action="store_true")

parser.add_argument("--dev", action="store_true")
parser.add_argument("--wandb_disabled", action="store_true")

args = parser.parse_args()

# Dev:
# python regression.py --model llama --task linear_regression --dev --wandb_disabled



######################## Hyperparameters ############################

if args.model == "llama":
    hp = {
        "llm_name": "Models/Meta-Llama-3-70B-Instruct",
        "api_key": VLLM_APIKEY,
        "base_url": VLLM_ENDPOINT,
    }

elif args.model == "gpt-4o":
    hp = {
        "llm_name": "gpt-4o",
        "api_key": OPENAI_API_KEY,
        "base_url": None,
    }

hp.update({
    "batch_size": args.batch_size,
    "eval_batch_size": args.eval_batch_size,
    "epochs": args.epochs,
    "task": args.task,
    "job_id": args.job_id,
    "model": args.model,
    "prior": None,
    "opt_name": "regression",
    "fx_name": "regression",
})

######################## Creating Dataset ############################

if hp['task'] == "linear_regression":
    m = 100
    X = 2 * np.random.rand(m, 1)
    y = 4 + 3 * X + np.random.randn(m, 1) * 0.1
    x_left = 0
    x_right = 2
    X_plot = np.linspace(x_left-1, x_right+1, 40)
    X_plot = np.expand_dims(X_plot, axis=1)
    y_plot = 4 + 3 * X_plot
    hp['opt_hp'] = "If the model is doing well, you can keep using the current descriptions. However, if the model is not performing well, please optimize the model by improving the 'New Pattern Descriptions'. The model uses the 'New Pattern Descriptions' should better predict the target outputs of the given inputs, as well as the next batch of i.i.d. input data from the same distribution. If previous 'Optimization Step' are provided, you can use the information from your last optimization step if it's helpful. DON'T use symbolic representation for the model!"
    if args.add_prior:
        hp['prior'] = "It looks like the data is generated from a simple and smooth function."
elif hp['task'] == "poly_regression":
    m = 100
    X = 4 * np.random.rand(m, 1) - 3
    y = 3 * X**2 + X + 2 + np.random.randn(m, 1)
    x_left = -3
    x_right = 1
    X_plot = np.linspace(x_left-2, x_right+2, 40)
    X_plot = np.expand_dims(X_plot, axis=1)
    y_plot = 3 * X_plot**2 + X_plot + 2
    hp['opt_hp'] = "If the model is doing well, you can keep using the current descriptions. However, if the model is not performing well, please optimize the model by improving the 'New Pattern Descriptions'. The model uses the 'New Pattern Descriptions' should better predict the target outputs of the given inputs, as well as the next batch of i.i.d. input data from the same distribution. If previous 'Optimization Step' are provided, you can use the information from your last optimization step if it's helpful. NOTE: both the model and you can only operate on the numerical precision of one decimal points!"
    if args.add_prior:
        hp['prior'] = "It looks like the data is generated from a simple and smooth function."
elif hp['task'] == "sine_regression":
    m = 100
    X = 6 * np.random.rand(m, 1) - 3
    y = np.sin(X) + np.random.randn(m, 1) * 0.01 + 2
    x_left = -3
    x_right = 3
    X_plot = np.linspace(x_left-3, x_right+3, 40)
    X_plot = np.expand_dims(X_plot, axis=1)
    y_plot = np.sin(X_plot) + 2
    hp['opt_hp'] = "If the model is doing well, you can keep using the current descriptions. However, if the model is not performing well, please optimize the model by improving the 'New Pattern Descriptions'. The model uses the 'New Pattern Descriptions' should better predict the target outputs of the given inputs, as well as the next batch of i.i.d. input data from the same distribution. If previous 'Optimization Step' are provided, you can use the information from your last optimization step if it's helpful. NOTE: both the model and you can only operate on the numerical precision of one decimal points!"
    if args.add_prior:
        hp['prior'] = "It looks like the data is generated by a periodic function."

# Round for numerical stability
X = np.round(X, 2)
y = np.round(y, 2)
X_plot = np.round(X_plot, 2)
y_plot = np.round(y_plot, 2)

# Splitting the dataset into training and testing
train_X = X
train_y = y

# Creating batches
train_X_batches, train_y_batches = create_batch(train_X, train_y, hp['batch_size'])

######################## Setting up model ############################
# Setup the API client
print(f"Setting up API client with base_url: {hp['base_url']} and api_key: {hp['api_key']}")
llm_endpoint, async_llm_endpoint = setup_api_client(hp['api_key'], hp['base_url'])

### Without any prior knowledge:
init_theta = "You are designed to do regression, i.e., to predict the output of any given input. Both input and output are real numbers."
if hp['prior'] is not None:
    init_theta += " " + hp['prior']
hp['init_theta'] = init_theta

######################## Setting Wandb ############################
tags = []
if args.dev:
    tags.append('dev_reg')
    project_name = "VML"
else:
    tags.append('prod_reg')
    project_name = "VML-Prod"


if hp["base_url"] is None:
    tags.append('openai')
elif "http://172.22." in hp['base_url']:
    tags.append('llama')
else:
    raise ValueError("Unknown API base_url")



if args.wandb_disabled:
    wandb_mode = "disabled"
else:
    wandb_mode = "online"

# Initialise wandb
while (
    True
):  # A workaround for the `wandb.errors.UsageError: Error communicating with wandb process`
    try:
        wandb.init(
            project=project_name,
            group=hp['task'],
            tags=tags,
            mode=wandb_mode,
        )
        break
    except:
        print("Retrying: wandb.init")
        time.sleep(5)
wandb.config.update(hp)

######################## Train ############################
train_losses = []
test_in_losses = []
test_out_losses = []
test_overall_losses = []

theta = init_theta
opt_state = None
theta_list = []
opt_state_list = []
theta_list.append(theta)
opt_state_list.append(opt_state)

# Training the model
y_plot_hat_all = []
for epoch in range(hp['epochs']):
    print(f"************ Epoch {epoch} ************")
    for i in range(len(train_X_batches)):
        print(f"************ Testing ************")
        y_plot_hat = test(X_plot, async_llm_endpoint, hp['llm_name'], hp['fx_name'], theta, hp['eval_batch_size'], verbose=False)
        y_plot_hat_all.append(y_plot_hat)
        in_loss, out_loss, overall_loss = calculate_mse_in_out_loss(X_plot, y_plot_hat, y_plot, x_left, x_right, verbose=True)
        test_in_losses.append(in_loss)
        test_out_losses.append(out_loss)
        test_overall_losses.append(overall_loss)
        plot_regression(train_X, train_y, X_plot, y_plot_hat, epoch*len(train_X_batches)+i)
        wandb.log({
            "test_in_loss": in_loss,
            "test_out_loss": out_loss,
            "test_overall_loss": overall_loss,
        })

        print(f"************ Training ************")
        theta, train_loss, opt_state = train_step(theta, train_X_batches, train_y_batches, 
                                                  llm_endpoint, async_llm_endpoint,
                                                  hp, epoch, i, 
                                                #   last_opt_state=None, 
                                                  last_opt_state=opt_state, 
                                                  verbose=True)
        train_losses.append(train_loss)
        theta_list.append(theta)
        opt_state_list.append(opt_state)

        theta = init_theta + " " + theta 
    
print(f"************ Final Testing ************")
y_plot_hat = test(X_plot, async_llm_endpoint, hp['llm_name'], hp['fx_name'], theta, hp['eval_batch_size'], verbose=False)
y_plot_hat_all.append(y_plot_hat)
in_loss, out_loss, overall_loss = calculate_mse_in_out_loss(X_plot, y_plot_hat, y_plot, x_left, x_right, verbose=True)
test_in_losses.append(in_loss)
test_out_losses.append(out_loss)
test_overall_losses.append(overall_loss)
plot_regression(train_X, train_y, X_plot, y_plot_hat, hp['epochs']*len(train_X_batches))

print("************ Training Losses ************")
print(train_losses)

print("************ Save results ************")
results = {
    "hp": hp,
    "train_X": train_X,
    "train_y": train_y,
    "X_plot": X_plot,
    "y_plot": y_plot,
    "y_plot_hat_all": y_plot_hat_all,
    "train_losses": train_losses,
    "test_in_losses": test_in_losses,
    "test_out_losses": test_out_losses,
    "test_overall_losses": test_overall_losses,

    "theta_list": theta_list,
    "opt_state_list": opt_state_list,
}

results_dir = "vml_results_regression"
os.makedirs(results_dir, exist_ok=True)
file_name = wandb.run.name
file_path = osp.join(results_dir, f"{file_name}.json")
with open(file_path, 'w') as f:
    json.dump(results, f, default=default)

print(f"************ Results saved to {file_path} ************")