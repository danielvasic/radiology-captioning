import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


# Encoder CNN
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # Remove the last fully-connected layer
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
    
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def generate(self, features, vocab, lengths):
        """Generate captions for given image features using greedy search, respecting lengths."""
        batch_size = features.size(0)  # Get the batch size
        max_length = max(lengths)  # Maximum length in the batch
        sampled_ids = [[] for _ in range(batch_size)]  # Create a separate list for each image in the batch
        stop_generation = [False] * batch_size  # Track whether to stop generating for each image
        inputs = features.unsqueeze(1)
        states = None

        # Iterate for a maximum number of time steps (max_length)
        for t in range(max_length):
            hiddens, states = self.lstm(inputs, states)  # Forward pass through LSTM
            outputs = self.linear(hiddens.squeeze(1))    # Compute outputs

            # Take the word with the maximum probability for each image in the batch
            _, predicted = outputs.max(1)  # Shape: [batch_size]

            # Append predicted words to each image's caption list, respecting individual lengths
            for i in range(batch_size):
                if len(sampled_ids[i]) < lengths[i] and not stop_generation[i]:
                    sampled_ids[i].append(predicted[i].item())
                    if predicted[i].item() == vocab.stoi["<end>"]:
                        stop_generation[i] = True

            # Embed the predicted words for the next input step
            inputs = self.embed(predicted)  # Shape: [batch_size, embed_size]
            inputs = inputs.unsqueeze(1)    # Add time step dimension

            # Stop early if all captions are complete
            if all(stop_generation):
                break

        return sampled_ids  # Return a list of word indices for each image in the batch

# Image Captioning Model
class BaselineCaptioninngModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(BaselineCaptioninngModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
    
    def forward(self, images, captions, lengths):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        return outputs