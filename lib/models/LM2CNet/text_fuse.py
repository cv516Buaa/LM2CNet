import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexWeightGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(ComplexWeightGenerator, self).__init__()
        self.num_heads = num_heads

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_heads * 2)

        self.residual_gate = nn.Linear(input_dim, 1)

    def forward(self, tensor1, tensor2):
        combined = torch.cat((tensor1.mean(dim=1), tensor2.mean(dim=1)), dim=-1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        weights = torch.sigmoid(self.fc3(x))

        weights = weights.view(-1, self.num_heads, 2)

        residual_weight = torch.sigmoid(self.residual_gate(combined))

        return weights, residual_weight


class DynamicWeightedFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=2):
        super(DynamicWeightedFusion, self).__init__()
        self.weight_generator = ComplexWeightGenerator(input_dim * 2, hidden_dim, num_heads)
        self.num_heads = num_heads
        self.input_dim = input_dim

    def forward(self, tensor, tensor2):

        weights, residual_weight = self.weight_generator(tensor, tensor2)

        fused_tensors = []

        for i in range(self.num_heads):
            weight1 = weights[:, i, 0:1].unsqueeze(1)
            weight1 = weight1.expand(-1, 110, 256)
            weight2 = weights[:, i, 1:2].unsqueeze(1)
            weight2 = weight2.expand(-1, 110, 256)
            fused_tensor = weight1 * tensor + weight2 * tensor2
            fused_tensors.append(fused_tensor)

        fused_tensor = torch.mean(torch.stack(fused_tensors, dim=0), dim=0)

        return fused_tensor


