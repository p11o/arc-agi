import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptiveFewShotGenerator(nn.Module):
    def __init__(self, max_seq_length, num_classes=10, d_model=512):
        super(AdaptiveFewShotGenerator, self).__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        # Encoder to get features from each input/output pair
        self.encoder = nn.Sequential(
            nn.Conv2d(num_classes, 64, kernel_size=3, padding=1),  # Assuming input_channels channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            # nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1))
        )
        
        self.flatten = nn.Flatten()
        
        # Transformer for sequence processing
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=512, dropout=0.1)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=6)
        
        self.content_decoder = nn.Linear(d_model, 30 * 30 * num_classes)  # Adjusted for 30x30 output

        # Mask creation for training 
        self.register_buffer('mask', torch.triu(torch.ones(max_seq_length, max_seq_length), diagonal=1).bool())
        self.pos_encoder = self._generate_positional_encoding(d_model, max_seq_length)

    def _generate_positional_encoding(self, d_model, max_len=5000):
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)  # Unsqueeze for batch dimension

    def forward(self, sequence):

        # sequence shape: [batch_size, seq_len, input_dim], where input_dim is the number of integers per sequence item
        # sequence = sequence.long()
        # Convert integers to one-hot encoding
        # sequence = F.one_hot(sequence, num_classes=self.num_classes).float()
        
        # Now sequence shape: [batch_size, seq_len, input_dim, num_classes]
        batch_size, seq_len, *_ = sequence.shape
        
        # Flatten sequence for CNN input: [batch_size * seq_len, num_classes * input_dim]
        sequence = sequence.view(-1, self.num_classes, 30, 30)
        
        # Encode each frame in the sequence
        encoded = self.encoder(sequence)
        encoded = encoded.view(batch_size, seq_len, self.d_model)
        encoded = encoded + self.pos_encoder[:, :seq_len, :].to(encoded.device)
        
        # Transformer layer
        context = self.transformer(encoded.transpose(0, 1), mask=self.mask[:seq_len, :seq_len].to(encoded.device)).transpose(0, 1)

        
        # Decode for content prediction - now outputting logits for all classes
        logits = self.content_decoder(context[:, -1, :]).view(batch_size, 30, 30, self.num_classes)  # Shape: [batch_size, num_classes]
        return logits
        # class_indices = torch.argmax(logits, dim=1)  # Shape: [batch_size, 30, 30]
        # If you want to return to an integer prediction, use argmax
        # However, typically you would return logits or softmax for loss calculation
        # return class_indices.float()



class RelationshipTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(RelationshipTransformer, self).__init__()
        
        # Embedding layers for A, B, and C
        self.embed_A = nn.Embedding(vocab_size, d_model)
        self.embed_B = nn.Embedding(vocab_size, d_model)
        self.embed_C = nn.Embedding(vocab_size, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Linear layer to predict the vocabulary size for output
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Layer to combine embeddings for relationship understanding
        self.relationship_layer = nn.Linear(d_model * 2, d_model)

    def encode(self, src):
        # Encode the source sequences
        return self.transformer_encoder(src)

    def decode(self, tgt, memory, relationship_vector):
        # Decode with additional relationship context
        return self.transformer_decoder(tgt, memory + relationship_vector.unsqueeze(1).repeat(1, tgt.size(1), 1))

    def forward(self, src_A, src_B, tgt):
        # Embed inputs
        embed_A = self.embed_A(src_A)
        embed_B = self.embed_B(src_B)
        embed_C = self.embed_C(tgt)
        
        # Combine A and B for encoding
        combined_src = torch.cat([embed_A, embed_B], dim=-1)
        relationship_vector = self.relationship_layer(combined_src.mean(dim=1))  # Simplistic relationship vector
        
        # Pass through encoder
        memory = self.encode(torch.stack([embed_A, embed_B], dim=1))
        
        # Decode with the relationship context
        output = self.decode(embed_C, memory, relationship_vector)
        
        # Final output transformation
        return self.output_layer(output)