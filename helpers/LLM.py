import os
from openai import OpenAI
import json

import google.generativeai as genai

openai_client = OpenAI()
GOOGLE_API_KEY=os.environ['GOOGLE_API_KEY']
gemini_client = genai.GenerativeModel('gemini-pro')

genai.configure(api_key=GOOGLE_API_KEY)

with open('helpers/prompt_template.txt', 'r') as f:
    prompt_templates = f.read()

def format_check(hyperparameters):
    """
    This function takes a dictionary of hyperparameters and returns a string.
    {
      "prompts": {
        "fantasy landscape": 1,
        "majestic creature": 0.5,
        "vibrant colors": 0.5
      },
      "beta": 0.1,
      "action_num": 4,
      "weighting": "tanh",
      "sample_step": 25,
      "tau": 0.1
    }
    """
    try:
        prompts = hyperparameters['prompts']
        beta = hyperparameters['beta']
        action_num = hyperparameters['action_num']
        weighting = hyperparameters['weighting']
        sample_step = hyperparameters['sample_step']
        tau = hyperparameters['tau']
        return True
    except:
        return False

def get_prompt(history):
    """
    This function takes a list of trajectories and returns a prompt for the GPT3 model.
    """
    history = sorted(history, key=lambda x: x['score'])[-150:]
    str_list = []
    for data in history:
        hyperparameters = data['hyperparameters']
        score = data['score']
        hyperparameters_json_string = json.dumps(hyperparameters, indent=2)
        str_list.append(f"Hyperparameters:\n{hyperparameters_json_string}\nScore: {score}\n\n\n")
    trajectories = '\n'.join(str_list)
    return prompt_templates.format(history=trajectories)

def opro_gpt(trajectories, model="gpt-3.5-turbo-0125"):
    """
    This function takes a list of trajectories and returns the optimal policy for the GPT model.
    """
    # Load the GPT4 model
    failed_time = 0
    while failed_time < 3:
      input = get_prompt(trajectories)
      response = openai_client.chat.completions.create(
        model=model,
        response_format={ "type": "json_object" },
        messages=[
          {"role": "system", "content": "You are a helpful assistant to tune the hyperparameters as json file of a sampling algorithm."},
          {"role": "user", "content": input}
        ],
        temperature=1.0
      )
      output = response.choices[0].message.content
      with open(f'API_hist/{response.created}.json', 'w') as f:
          data = {
              'input': input,
              'output': output,
              'model': model,
          }
          json.dump(data, f, indent=4)
      if format_check(json.loads(output)): 
          return json.loads(output)
    
    raise Exception("Failed to get a valid response from the GPT model.")

def opro_gemini(trajectories):
    response = gemini_client.generate_content(get_prompt(trajectories))
    return response.text
    
if __name__ == "__main__":
    for_test = [{
        "hyperparameters": {
            "prompts": {
                "good image": 1,
                "bad anatomy": -1
            },
            "beta": 0.1,
            "action_num": 8,
            "weighting": "tanh",
            "sample_step": 25,
            "tau": 0.1
        },
        "score": 8
    }, {
        "hyperparameters": {
            "prompts": {
                "good image": 1,
                "bad anatomy": -1
            },
            "beta": 0.1,
            "action_num": 16,
            "weighting": "tanh",
            "sample_step": 25,
            "tau": 0.1
        },
        "score": 6
    },
    {
        "hyperparameters": {
            "prompts": {
                "good image": 1,
                "bad anatomy": -1
            },
            "beta": 0.1,
            "action_num": 32,
            "weighting": "tanh",
            "sample_step": 25,
            "tau": 0.1
        },
        "score": 4
    }
    ]
    print(opro_gpt(for_test))