import torch
import torch.nn as nn

from arc.models import rule_extractor, rule_applier
from arc.loader import dataloader

# Assuming these are defined correctly
criterion = nn.MSELoss()
optimizer_extractor = torch.optim.Adam(rule_extractor.parameters())
optimizer_applier = torch.optim.Adam(rule_applier.parameters())

num_epochs = 10
output_size = 100  # Ensure this matches your model's output size

for epoch in range(num_epochs):
    rule_extractor.train()
    rule_applier.train()

        
    # Rest of your training loop...
    for batch_input, batch_output, test_input, test_output in dataloader:
        optimizer_extractor.zero_grad()
        optimizer_applier.zero_grad()
        
        # Apply rule
        combined_input = torch.cat([batch_input, batch_output.unsqueeze(1)], dim=1)
        # Extract rule        
        rule_embedding = rule_extractor(combined_input)
        
        # Ensure shapes match for loss calculation
        if predicted_output.shape != test_output.shape:
            # Example adjustment, adjust according to your actual needs
            predicted_output = predicted_output.view(test_output.shape)
        
        loss = criterion(predicted_output, test_output)
        loss.backward()
        optimizer_extractor.step()
        optimizer_applier.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Optional: Validation step here if you have a validation set

# Optionally, switch to eval mode for testing or inference
# rule_extractor.eval()
# rule_applier.eval()