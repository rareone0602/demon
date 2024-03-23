import json

with open('hyperparameters_list.jsonl', 'r') as f:
    hyperparameters_list = f.read().splitlines()
with open('hyperparameters_list_original.jsonl', 'w') as f:
    for data in [json.loads(x) for x in hyperparameters_list]:
        hyperparameters = data["hyperparameters"]
        score = data['score'] + (len(hyperparameters["prompts"]) + 1) * hyperparameters["action_num"] * hyperparameters["sample_step"] / 1000
        f.write(json.dumps({"hyperparameters": hyperparameters, "score": score}) + '\n')
    
