import torch
import torch.nn as nn

def conv_bn_relu6(in_channels, out_channels, kernel_size, stride):
    """A standard block of Conv1d -> BatchNorm -> ReLU6."""
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU6(inplace=True)
    )

class Bottleneck(nn.Module):
    """Inverted Residual Bottleneck Block from MobileNetV2."""
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(Bottleneck, self).__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio

        # Check if to use residual connection
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        # Expansion layer (1x1 Pointwise Conv)
        if expand_ratio != 1:
            layers.append(conv_bn_relu6(in_channels, hidden_dim, kernel_size=1, stride=1))

        # Depthwise convolution
        layers.extend([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])

        # Projection layer (1x1 Pointwise Conv)
        layers.extend([
            nn.Conv1d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNet1D(nn.Module):
    """A 1D MobileNet-style model based on the provided C code."""
    def __init__(self, num_classes=5, in_channels=1, input_length=1000):
        super(MobileNet1D, self).__init__()

        # Configuration for the 10 bottleneck blocks seen in the C code
        # Format: (t, c, s) -> (expansion_ratio, out_channels, stride)
        bottleneck_configs = [
            (1, 16, 1),
            (6, 24, 2),
            (6, 24, 1),
            (6, 32, 2),
            (6, 32, 1),
            (6, 32, 1),
            (6, 64, 2),
            (6, 64, 1),
            (6, 64, 1),
            (6, 96, 1),
        ]

        # Initial convolution layer
        self.initial_conv = conv_bn_relu6(in_channels, 32, kernel_size=3, stride=2)
        current_channels = 32

        # Build bottleneck layers
        bottleneck_layers = []
        for t, c, s in bottleneck_configs:
            bottleneck_layers.append(Bottleneck(current_channels, c, s, expand_ratio=t))
            current_channels = c
        self.bottlenecks = nn.Sequential(*bottleneck_layers)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(current_channels, num_classes)
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.bottlenecks(x)
        x = self.classifier(x)
        return x

def create_mobilenet_1d(num_classes=5):
    """Function to create an instance of the MobileNet1D model."""
    return MobileNet1D(num_classes=num_classes)