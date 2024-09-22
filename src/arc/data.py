import json


def read(path: str):
    with open(path, 'r') as f:
        return json.load(f)
    
challenges = read('/data/arc-agi_training_challenges.json')
solutions = read('/data/arc-agi_training_solutions.json')
