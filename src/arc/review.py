import torch
import data
import models

state = torch.load('/data/arc-agi.pth')

model = models.AdaptiveFewShotGenerator(max_seq_length=101).to('cpu')
model.load_state_dict(state)
model.eval()

df = data.to_df(data.challenges, data.solutions)
ds = data.FewShotDataset(df, 101)

if __name__ == '__main__':
    expected = ds[0][-1]
    actual = model(ds[0].unsqueeze(0))
    print("Expected")
    print(expected)
    print("Actual")
    print(actual)