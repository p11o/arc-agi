import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveFewShotGenerator(nn.Module):
    def __init__(self):
        super(AdaptiveFewShotGenerator, self).__init__()
        
        # Encoder for both input and output
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Assuming 1 channel for simplicity
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1))  # This will output a 1x1 spatial dimension
        )
        
        # Flatten layer after adaptive pooling
        self.flatten = nn.Flatten()
        
        # Transformer for context understanding
        self.processor = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.transformer = nn.TransformerEncoder(self.processor, num_layers=2)
        
        # Decoder to predict the output size and then reshape
        self.size_predictor = nn.Linear(128, 2)  # Predict height and width
        self.content_decoder = nn.Linear(128, 81)  # Max size we'll handle, can be adjusted

    def forward(self, prompt_set, challenge_input, input_sizes, output_sizes):
        # prompt_set: List of [input, output] pairs, challenge_input: New input tensor
        # input_sizes, output_sizes: List of tuples representing (height, width)
        
        # Encode all inputs and outputs
        encoded_prompts = []
        for pair, in_size, out_size in zip(prompt_set, input_sizes, output_sizes):
            # Combine input and output, reshape to 4D tensor for convolution
            combined = torch.cat([pair[0].unsqueeze(0), pair[1].unsqueeze(0)], dim=0).unsqueeze(0)
            encoded = self.encoder(combined)
            encoded_prompts.append(self.flatten(encoded))
        
        # Encode challenge input
        challenge_encoded = self.flatten(self.encoder(challenge_input.unsqueeze(0).unsqueeze(0)))
        encoded_sequence = torch.stack(encoded_prompts + [challenge_encoded])
        
        # Create mask for transformer (to ignore padding if any)
        mask = self.create_mask(encoded_sequence, input_sizes + [input_sizes[-1]])
        
        # Process through transformer
        context = self.transformer(encoded_sequence.unsqueeze(1), src_key_padding_mask=mask).squeeze(1)
        
        # Predict size of output
        predicted_size = torch.sigmoid(self.size_predictor(context[-1])) * 9  # Assuming max size is 9x9
        h, w = torch.round(predicted_size).int()
        
        # Generate content
        content = self.content_decoder(context[-1]).view(h, w)
        
        return content

    def create_mask(self, sequence, sizes):
        mask = torch.zeros((sequence.size(0), sequence.size(0)), dtype=torch.bool)
        for i, size in enumerate(sizes):
            mask[i, len(sizes) - (i < len(sizes))] = True  # Masking out the challenge input in prompt
        return mask

# Usage would involve creating tensors of varying sizes, keeping track of these sizes, 
# and passing them into the model along with the tensors.