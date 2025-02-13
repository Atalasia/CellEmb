import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, z_i, z_j, pccs, data_count):
        """
        z_i, z_j: Embeddings of the data pairs (batch_size x embedding_dim)
        labels: Binary labels (batch_size), 1 for positive pairs, 0 for negative pairs
        """
        # Compute Euclidean distance between embeddings
        distances = torch.norm(z_i - z_j, dim=1)

        # Positive pairs loss
        positive_loss = pccs * torch.pow(distances, 2)

        # Negative pairs loss
        negative_loss = (1 - pccs) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)

        # Combine losses and average over the batch
        dc_multiplier = torch.log(data_count)
        loss = dc_multiplier * (positive_loss + negative_loss)

        return loss