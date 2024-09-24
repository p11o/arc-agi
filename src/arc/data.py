import json
import pandas as pd


def read(path: str):
    with open(path, 'r') as f:
        return json.load(f)
    
challenges = read('/data/arc-agi_training_challenges.json')
solutions = read('/data/arc-agi_training_solutions.json')

def to_df(challenges, solutions):
    """
    id: {group} + {group}.train|test.idx
    group_id: {group}
    type: test|train
    input: {group} | (.train[], .test[]) | .input
    output: {group} | (.train[].output) 
    """
    df_array = []
    for group_id, group in challenges.items():
        data = [*group["train"], *[{**test, "type": "test", "output": solutions[group_id][i]} for i, test in enumerate(group["test"])]]
        for i, datum in enumerate(data):
            _input, _output = datum["input"], datum["output"]

            df_array.append([f"{group_id}{i}", group_id, datum.get("type", "train"), pd.array(_input), pd.array(_output)])

    return pd.DataFrame(df_array, columns=["id", "group_id", "type", "input", "output"])
