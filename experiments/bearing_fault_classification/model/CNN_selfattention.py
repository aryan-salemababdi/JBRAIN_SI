import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv1d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, L = x.size()
        Q = self.query_conv(x).permute(0, 2, 1)
        K = self.key_conv(x)
        energy = torch.bmm(Q, K)
        attention = F.softmax(energy / ((C / 8) ** 0.5), dim=-1)
        V = self.value_conv(x)
        out = torch.bmm(V, attention.permute(0, 2, 1))
        out = self.gamma * out + x
        return out


class CNN_SelfAttention(nn.Module):
    def __init__(self, num_classes=10, input_length=None):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(1, 32, kernel_size=16, padding=8)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=8, padding=4)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool  = nn.MaxPool1d(4)
        self.attn  = SelfAttention(64)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        if input_length is not None:
            # Create a dummy input tensor with shape (batch_size=1, channels=1, length=input_length)
            dummy_input = torch.zeros(1, 1, input_length)
            with torch.no_grad():
                dummy_features = self.forward_features(dummy_input)
            fc_input_dim = dummy_features.shape[1]
        else:
            # fallback in case input_length is not provided
            fc_input_dim = 64 * 19  
            
        # ---- Fully Connected layers ----
        self.fc1 = nn.Linear(fc_input_dim, 256)
        self.fc2 = nn.Linear(256, self.num_classes)

    def forward_features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.attn(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
