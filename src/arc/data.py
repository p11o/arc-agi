import json
import pandas as pd
import numpy as np


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

            df_array.append([f"{group_id}{i}", group_id, datum.get("type", "train"), np.array(_input), np.array(_output)])

    return pd.DataFrame(df_array, columns=["id", "group_id", "type", "input", "output"])


def rotate_90(row):
    new_rows = []
    # Rotate 90, 180, and 270 degrees
    for k in range(1, 4):  # k=1 is 90°, k=2 is 180°, k=3 is 270°
        new_row = row.copy()
        new_row['id'] = f"{new_row['id']}{k}"
        new_row['type'] = 'train'
        new_row['input'] = np.rot90(new_row['input'], k)
        new_row['output'] = np.rot90(new_row['output'], k)
        
        new_rows.append(new_row)
    return new_rows

def _rotate_color(x: int, by: int = 1, mod: int = 10):
    if x == 0:
        return 0
    x = (x + by) % mod
    if x <= by:
        return x + 1
    return x

_v_rotate_color = np.vectorize(_rotate_color, otypes=[np.uint8])

def rotate_color(row):
    new_rows = []
    for i in range(1, 9): # 1 -> 9 because we skip 0's since they are blank
        new_row = row.copy()
        new_row['id'] = f"{new_row['id']}{i}"
        new_row['type'] = 'train'
        new_row['input'] = _v_rotate_color(new_row['input'], i)
        new_row['output'] = _v_rotate_color(new_row['output'], i)
        new_rows.append(new_row)
    return new_rows

def augment_data(df):
    # New DataFrame to hold augmented data
    rows = [row for _, row in df.iterrows()]

    for fn in [rotate_90, rotate_color]:
        new_rows = []
        for row in rows:
            for r in fn(row):
                new_rows.append(r)
        rows = [*rows, *new_rows]
    return pd.DataFrame(rows)
