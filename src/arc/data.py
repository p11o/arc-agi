import json
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset



def read(path: str):
    with open(path, 'r') as f:
        return json.load(f)
    
challenges = read('/data/arc-agi_training_challenges.json')
solutions = read('/data/arc-agi_training_solutions.json')

test_challenges = read('/data/arc-agi_evaluation_challenges.json')
test_solutions = read('/data/arc-agi_evaluation_solutions.json')

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
    for k in range(1, 4):
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
    for i in range(1, 9): # 1 -> 9 because we skip 0's and the current value
        new_row = row.copy()
        _id = new_row['id']
        new_row['id'] = f"{_id}{i}"
        new_row['group_id'] = _id
        new_row['type'] = 'train'
        new_row['input'] = _v_rotate_color(new_row['input'], i)
        new_row['output'] = _v_rotate_color(new_row['output'], i)
        new_rows.append(new_row)
    return new_rows

def augment_data(df):
    rows = [row for _, row in df.iterrows()]

    # for fn in [rotate_90]:
    for fn in [rotate_90, rotate_color]:
        new_rows = []
        for row in rows:
            for r in fn(row):
                new_rows.append(r)
        rows = [*rows, *new_rows]
    return pd.DataFrame(rows)


# Assuming `convert_to_rgb` function is defined as previously
COLORS = [
    [0, 0, 0],       # 0 - Black
    [255, 0, 0],     # 1 - Red
    [0, 255, 0],     # 2 - Green
    [0, 0, 255],     # 3 - Blue
    [255, 255, 0],   # 4 - Yellow
    [255, 165, 0],   # 5 - Orange
    [128, 0, 128],   # 6 - Purple
    [0, 255, 255],   # 7 - Cyan
    [255, 192, 203], # 8 - Pink
    [128, 128, 128],  # 9 - Grey
]

def convert_to_rgb(matrix):
    if not (1 <= matrix.shape[0] <= 30 and 1 <= matrix.shape[1] <= 30):
        raise ValueError("Matrix dimensions must be between 1x1 and 30x30")
    if not np.all(np.isin(matrix, range(10))):
        raise ValueError("All values in the matrix must be integers from 0 to 9")
    
    output = np.array(COLORS)[matrix]
    return output.astype(np.uint8)


def prepare_for_vgg(matrix):
    if len(matrix.shape) != 2:
        raise ValueError("Matrix should be 2d")
    matrix = convert_to_rgb(matrix)

    # Ensure the input is in the correct format
    if matrix.shape[2] != 3:
        raise ValueError("RGB matrix should have 3 channels")

    if matrix.shape[0] != 224 or matrix.shape[1] != 224:
        top = bottom = left = right = 0
    if matrix.shape[0] < 224:
        top = (224 - matrix.shape[0]) // 2
        bottom = 224 - matrix.shape[0] - top
    if matrix.shape[1] < 224:
        left = (224 - matrix.shape[1]) // 2
        right = 224 - matrix.shape[1] - left
    padded = np.pad(matrix, ((top, bottom), (left, right), (0, 0)), 'constant', constant_values=0)

    # Convert to tensor from np HWC to tensor CHW (i.e. put channel first)
    tensor = torch.from_numpy(padded.transpose((2, 0, 1))).float()
    
    # Normalize
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor = normalize(tensor / 255.0)
        
    return tensor.unsqueeze(0)


# class FewShotDataset(Dataset):
#     def pad_to_size(self, tensor, target_size=(30, 30)):
#         current_size = tensor.shape[-2:]  # Get the last two dimensions which should be height and width
#         pad_h, pad_w = max(target_size[0] - current_size[0], 0), max(target_size[1] - current_size[1], 0)
        
#         # Calculate padding for each side (top, bottom, left, right)
#         pad = (0, pad_w, 0, pad_h)  # PyTorch padding format: (left, right, top, bottom)
        
#         # If no padding is necessary, F.pad will not change the tensor
#         padded_tensor = F.pad(tensor, pad, "constant", 0)  # 0 is the pad value, you can change this
        
#         return padded_tensor
    
#     def __init__(self, dataframe, max_length, num_classes=10):
#         self.data = dataframe
#         self.sequences = []
#         for group_id in self.data['group_id'].unique():
#             group = self.data[self.data['group_id'] == group_id]
#             sequence = []
#             for _, row in group.iterrows():
#                 # Convert input to tensor and resize to 30x30 if necessary
#                 input_tensor = torch.tensor(row['input'].copy(), dtype=torch.long)
#                 input_tensor = self.pad_to_size(input_tensor)
                
#                 # Convert output to tensor and resize to 30x30 if necessary
#                 output_tensor = torch.tensor(row['output'].copy(), dtype=torch.long)
#                 output_tensor = self.pad_to_size(output_tensor)
                
#                 sequence.extend([input_tensor, output_tensor])

#             # Pad sequence with 30x30 zero tensors to ensure a fixed length for batching
#             pad_length = max_length - len(sequence)
#             assert pad_length >= 0
#             zero_tensor = torch.zeros((30, 30), dtype=torch.long)
#             sequence = [*sequence[:-1], *([zero_tensor] * pad_length), sequence[-1]]
#             sequence = torch.stack(sequence)
#             sequence = F.one_hot(sequence, num_classes=num_classes).float()
#             self.sequences.append(sequence)

#     def __len__(self):
#         return len(self.sequences)

#     def __getitem__(self, idx):
#         return self.sequences[idx]
    

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

class FewShotDataset(Dataset):
    def pad_to_size(self, tensor, target_size=(30, 30)):
        current_size = tensor.shape[-2:]
        if current_size == target_size:
            return tensor
        
        pad_h = max(target_size[0] - current_size[0], 0)
        pad_w = max(target_size[1] - current_size[1], 0)
        pad = (0, pad_w, 0, pad_h)  # Padding format: (left, right, top, bottom)
        
        # Use a pre-allocated zero tensor for padding
        zero_pad = torch.zeros(target_size, dtype=tensor.dtype)
        zero_pad[:current_size[0], :current_size[1]] = tensor
        return zero_pad

    def __init__(self, dataframe, max_length, num_classes=10):
        self.data = dataframe
        self.max_length = max_length
        self.num_classes = num_classes
        self.group_ids = list(dataframe['group_id'].unique())
        self.zero_tensor = torch.zeros(30, 30, dtype=torch.long)
        self.sequences = []

        for group_id in self.data['group_id'].unique():
            group = self.data[self.data['group_id'] == group_id]
            self.sequences.append(self.create_sequence(group))

    def __len__(self):
        return len(self.group_ids)

    def __getitem__(self, idx):
        return self.sequences[idx]

    def create_sequence(self, group):
        sequence = []
        for _, row in group.iterrows():
            input_tensor = self.process_tensor(row['input'])
            output_tensor = self.process_tensor(row['output'])
            sequence.extend([input_tensor, output_tensor])

        sequence = self.pad_sequence(sequence)
        return F.one_hot(torch.stack(sequence), num_classes=self.num_classes).float()

    def process_tensor(self, data):
        tensor = torch.tensor(data.copy(), dtype=torch.long)
        return self.pad_to_size(tensor)

    def pad_sequence(self, sequence):
        current_length = len(sequence)
        if current_length < self.max_length:
            pad_length = self.max_length - current_length
            # Use pre-allocated zero tensor for efficiency
            sequence.extend([self.zero_tensor] * pad_length)
        elif current_length > self.max_length:
            sequence = sequence[:self.max_length]  # Truncate if too long
        
        # Ensure last element is included and pad to match structure in original code
        if current_length < self.max_length:
            sequence[-1] = sequence[-1]  # Just ensuring the last element is correctly set
        
        return sequence
