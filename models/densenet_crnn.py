import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

class DenseNetCRNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        base = densenet121(weights=DenseNet121_Weights.DEFAULT)

        self.features = base.features
        
        self.features.pool0 = nn.Identity()
        self.features.transition3.pool = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d((1, None))

        self.rnn = nn.LSTM(
            input_size=1024,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = x.squeeze(2)

        x = x.permute(0, 2, 1)

        x, _ = self.rnn(x)

        x = self.fc(x)

        return x.permute(1, 0, 2)