import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class ARGDataset(Dataset):
    def __init__(self, challenges, solutions, max_size=(30, 30)):
        self.data = []
        self.max_size = max_size
        for key, value in challenges.items():
            for train_example in value['train']:
                self.data.append((
                    torch.tensor(train_example['input'], dtype=torch.long),
                    torch.tensor(train_example['output'], dtype=torch.long),
                    torch.tensor(value['test'][0]['input'], dtype=torch.short),
                    torch.tensor(solutions[key][0], dtype=torch.long)
                ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data, output_data, test_input, test_output = self.data[idx]
        return (
            self._pad_tensor(input_data),
            self._pad_tensor(output_data),
            self._pad_tensor(test_input),
            test_output  # Assuming test_output doesn't need padding, adjust if necessary
        )

    def _pad_tensor(self, tensor):
        if tensor.size() == torch.Size(self.max_size):
            return tensor
        else:
            padded = torch.zeros(self.max_size, dtype=tensor.dtype)
            padded[:tensor.size(0), :tensor.size(1)] = tensor
            return padded

def pad_collate(batch):
    (inputs, outputs, test_inputs, test_outputs) = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    outputs = pad_sequence(outputs, batch_first=True, padding_value=0)
    test_inputs = pad_sequence(test_inputs, batch_first=True, padding_value=0)
    # Pad test_outputs to the largest size in the batch
    max_test_output_size = max(output.size() for output in test_outputs)
    padded_test_outputs = []
    for to in test_outputs:
        pad = [0, max_test_output_size[1] - to.size(1), 0, max_test_output_size[0] - to.size(0)]
        padded_test_outputs.append(torch.nn.functional.pad(to, pad, value=0))
    return inputs, outputs, test_inputs, torch.stack(padded_test_outputs)



def read(path: str):
    with open(path, 'r') as f:
        return json.load(f)
    
challenges = read('/data/arc-agi_training_challenges.json')
solutions = read('/data/arc-agi_training_solutions.json')

dataset = ARGDataset(challenges, solutions)


# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)
