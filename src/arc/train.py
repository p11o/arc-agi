import models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MAX_SEQ_LENGTH = 101

VOCAB_SIZE = 10  # Example vocabulary size
D_MODEL = 512
NHEAD = 8
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1


# Model, Loss, Optimizer
# model = models.AdaptiveFewShotGenerator(max_seq_length=MAX_SEQ_LENGTH).to(DEVICE)
model = models.RelationshipTransformer(d_model=D_MODEL, vocab_size=VOCAB_SIZE)
criterion = nn.CrossEntropyLoss()  # Mean Squared Error for regression-like task of predicting pixel values
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# DataLoaders

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc="Training"):
        sequences = batch.to(device)
        # print("Shape of sequences:", sequences.shape)
        optimizer.zero_grad()
        
        outputs = model(sequences)
        # Assuming the ground truth for the last item's output is at sequences[:, -1, 1] or similar
        # ground_truth = sequences[:, -1, 1].view(-1, 30, 30).to(device)  # Adjust this based on how your data is structured
        ground_truth = sequences[:, -1, :, :].to(device)
        loss = criterion(outputs, ground_truth)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            sequences = batch.to(device)
            outputs = model(sequences)
            # ground_truth = sequences[:, -1, 1].view(-1, 30, 30).to(device)  # Adjust as necessary
            ground_truth = sequences[:, -1, :, :].to(device)

            loss = criterion(outputs, ground_truth)
            running_loss += loss.item()
    
    return running_loss / len(val_loader)

# Training Loop
def run(train_loader, val_loader):
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        torch.save(model.state_dict(), '/data/arc-agi.pth')
        
        # Here you might want to add model checkpoint saving, early stopping, or learning rate scheduling
    
    # Save the model at the end if desired
    # torch.save(model.state_dict(), '/data/arc-agi.pth')


if __name__ == '__main__':
    import data
    train_df = data.to_df(data.challenges, data.solutions)
    train_df = data.augment_data(train_df)
    test_df = data.to_df(data.test_challenges, data.test_solutions)

    # Create datasets
    train_dataset = data.FewShotDataset(train_df, MAX_SEQ_LENGTH)
    test_dataset = data.FewShotDataset(test_df, MAX_SEQ_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    run(train_loader, val_loader)
