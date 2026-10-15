"""Feed-forward neural-network classifiers for SGA prediction."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class FNNClassifierTri3(nn.Module):
    """Large four-layer classifier (in -> h -> 2h -> h -> 1)."""

    def __init__(self, input_size, dropout_rate, layer_output_size,
                 pretrained_network=None, freeze_all_layers=False):
        super().__init__()
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        self.layer_output_size = layer_output_size
        self.freeze_all_layers = freeze_all_layers

        self.layer1 = nn.Linear(self.input_size, self.layer_output_size)
        self.layer2 = nn.Linear(self.layer_output_size, self.layer_output_size * 2)
        self.layer3 = nn.Linear(self.layer_output_size * 2, self.layer_output_size)
        self.layer4 = nn.Linear(self.layer_output_size, 1)

        self.dropout = nn.Dropout(p=self.dropout_rate)

        if pretrained_network is not None:
            self.layer1.load_state_dict(pretrained_network.layer1.state_dict())
            self.layer2.load_state_dict(pretrained_network.layer2.state_dict())
            self.layer3.load_state_dict(pretrained_network.layer3.state_dict())
            self.layer4.load_state_dict(pretrained_network.layer4.state_dict())
            if self.freeze_all_layers:
                freeze_layers = [self.layer1.parameters(), self.layer2.parameters(),
                                 self.layer3.parameters(), self.layer4.parameters()]
            else:
                freeze_layers = [self.layer1.parameters(), self.layer2.parameters(),
                                 self.layer3.parameters()]

            for param in freeze_layers:
                for p in param:
                    p.requires_grad = False

    def forward_layer(self, layer, x):
        """Apply one linear layer followed by Mish and dropout."""
        x = layer(x)
        x = F.mish(x)
        x = self.dropout(x)
        return x

    def get_weights(self):
        """Return the weight matrix of every linear layer as NumPy arrays."""
        weights = {}
        weights['layer1'] = self.layer1.weight.detach().numpy()
        weights['layer2'] = self.layer2.weight.detach().numpy()
        weights['layer3'] = self.layer3.weight.detach().numpy()
        weights['layer4'] = self.layer4.weight.detach().numpy()
        return weights

    def get_x_after_first_layer(self, x):
        """Representation after the first hidden block."""
        return self.forward_layer(self.layer1, x)

    def get_x_after_second_layer(self, x):
        """Representation after the second hidden block."""
        x_after_first_layer = self.get_x_after_first_layer(x)
        return self.forward_layer(self.layer2, x_after_first_layer)

    def get_x_after_third_layer(self, x):
        """Representation after the third hidden block."""
        x_after_second_layer = self.get_x_after_second_layer(x)
        return self.forward_layer(self.layer3, x_after_second_layer)

    def forward(self, x):
        """Return the raw logit for each row of ``x``."""
        x_after_third_layer = self.get_x_after_third_layer(x)
        return self.layer4(x_after_third_layer)


class FNNClassifierTri3_Medium(nn.Module):
    """Medium three-layer classifier (in -> h -> 2h -> 1)."""

    def __init__(self, input_size, dropout_rate, layer_output_size,
                 pretrained_network=None, freeze_all_layers=False):
        super().__init__()
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        self.layer_output_size = layer_output_size
        self.freeze_all_layers = freeze_all_layers

        self.layer1 = nn.Linear(self.input_size, self.layer_output_size)
        self.layer2 = nn.Linear(self.layer_output_size, self.layer_output_size * 2)
        self.layer3 = nn.Linear(self.layer_output_size * 2, 1)

        self.dropout = nn.Dropout(p=self.dropout_rate)

        if pretrained_network is not None:
            self.layer1.load_state_dict(pretrained_network.layer1.state_dict())
            self.layer2.load_state_dict(pretrained_network.layer2.state_dict())
            self.layer3.load_state_dict(pretrained_network.layer3.state_dict())
            if self.freeze_all_layers:
                freeze_layers = [self.layer1.parameters(), self.layer2.parameters(),
                                 self.layer3.parameters()]
            else:
                freeze_layers = [self.layer1.parameters(), self.layer2.parameters()]

            for param in freeze_layers:
                for p in param:
                    p.requires_grad = True

    def forward_layer(self, layer, x):
        """Apply one linear layer followed by Mish and dropout."""
        x = layer(x)
        x = F.mish(x)
        x = self.dropout(x)
        return x

    def get_weights(self):
        """Return the weight matrix of every linear layer as NumPy arrays."""
        weights = {}
        weights['layer1'] = self.layer1.weight.detach().numpy()
        weights['layer2'] = self.layer2.weight.detach().numpy()
        weights['layer3'] = self.layer3.weight.detach().numpy()
        return weights

    def get_x_after_first_layer(self, x):
        """Representation after the first hidden block."""
        return self.forward_layer(self.layer1, x)

    def get_x_after_second_layer(self, x):
        """Representation after the second hidden block."""
        x_after_first_layer = self.get_x_after_first_layer(x)
        return self.forward_layer(self.layer2, x_after_first_layer)

    def forward(self, x):
        """Return the raw logit for each row of ``x``."""
        x_after_second_layer = self.get_x_after_second_layer(x)
        return self.layer3(x_after_second_layer)


class FNNClassifierTri3_Small(nn.Module):
    """Small two-layer classifier (in -> 4 -> 1)."""

    def __init__(self, input_size, dropout_rate, layer_output_size,
                 pretrained_network=None, freeze_all_layers=False):
        super().__init__()
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        self.layer_output_size = layer_output_size
        self.freeze_all_layers = freeze_all_layers

        self.layer1 = nn.Linear(self.input_size, 4)
        self.layer2 = nn.Linear(4, 1)

        self.dropout = nn.Dropout(p=self.dropout_rate)

        if pretrained_network is not None:
            self.layer1.load_state_dict(pretrained_network.layer1.state_dict())
            self.layer2.load_state_dict(pretrained_network.layer2.state_dict())
            if self.freeze_all_layers:
                freeze_layers = [self.layer1.parameters(), self.layer2.parameters()]
            else:
                freeze_layers = [self.layer1.parameters()]

            for param in freeze_layers:
                for p in param:
                    p.requires_grad = True

    def forward_layer(self, layer, x):
        """Apply one linear layer followed by Mish and dropout."""
        x = layer(x)
        x = F.mish(x)
        x = self.dropout(x)
        return x

    def get_weights(self):
        """Return the weight matrix of every linear layer as NumPy arrays."""
        weights = {}
        weights['layer1'] = self.layer1.weight.detach().numpy()
        weights['layer2'] = self.layer2.weight.detach().numpy()
        return weights

    def get_x_after_first_layer(self, x):
        """Representation after the first hidden block."""
        return self.forward_layer(self.layer1, x)

    def forward(self, x):
        """Return the raw logit for each row of ``x``."""
        x_after_first_layer = self.get_x_after_first_layer(x)
        return self.layer2(x_after_first_layer)


class FNNClassifierTri3_Test(nn.Module):
    """Wide four-layer variant (in -> 8*in -> 8*in -> 2*in -> 1)."""

    def __init__(self, input_size, dropout_rate, layer_output_size,
                 pretrained_network=None, freeze_all_layers=False):
        super().__init__()
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        self.layer_output_size = layer_output_size
        self.freeze_all_layers = freeze_all_layers

        self.layer1 = nn.Linear(self.input_size, self.input_size * 8)
        self.layer2 = nn.Linear(self.input_size * 8, self.input_size * 8)
        self.layer3 = nn.Linear(self.input_size * 8, self.input_size * 2)
        self.layer4 = nn.Linear(self.input_size * 2, 1)

        self.dropout = nn.Dropout(p=self.dropout_rate)

        if pretrained_network is not None:
            self.layer1.load_state_dict(pretrained_network.layer1.state_dict())
            self.layer2.load_state_dict(pretrained_network.layer2.state_dict())
            self.layer3.load_state_dict(pretrained_network.layer3.state_dict())
            if self.freeze_all_layers:
                freeze_layers = [self.layer1.parameters(), self.layer2.parameters(),
                                 self.layer3.parameters(), self.layer4.parameters()]
            else:
                freeze_layers = [self.layer1.parameters(), self.layer2.parameters(),
                                 self.layer3.parameters()]

            for param in freeze_layers:
                for p in param:
                    p.requires_grad = False

    def forward_layer(self, layer, x):
        """Apply one linear layer followed by Mish and dropout."""
        x = layer(x)
        x = F.mish(x)
        x = self.dropout(x)
        return x

    def get_weights(self):
        """Return the weight matrix of every linear layer as NumPy arrays."""
        weights = {}
        weights['layer1'] = self.layer1.weight.detach().numpy()
        weights['layer2'] = self.layer2.weight.detach().numpy()
        weights['layer3'] = self.layer3.weight.detach().numpy()
        weights['layer4'] = self.layer4.weight.detach().numpy()
        return weights

    def get_x_after_first_layer(self, x):
        """Representation after the first hidden block."""
        return self.forward_layer(self.layer1, x)

    def get_x_after_second_layer(self, x):
        """Representation after the second hidden block."""
        x_after_first_layer = self.get_x_after_first_layer(x)
        return self.forward_layer(self.layer2, x_after_first_layer)

    def get_x_after_third_layer(self, x):
        """Representation after the third hidden block."""
        x_after_second_layer = self.get_x_after_second_layer(x)
        return self.forward_layer(self.layer3, x_after_second_layer)

    def forward(self, x):
        """Return the raw logit for each row of ``x``."""
        x_after_third_layer = self.get_x_after_third_layer(x)
        return self.layer4(x_after_third_layer)


class FNNClassifierTri3_Calibration(nn.Module):
    """Two-layer classifier preceded by a learnable per-feature calibration layer."""

    def __init__(self, input_size, dropout_rate, layer_output_size,
                 pretrained_network=None, freeze_all_layers=True):
        super().__init__()
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        self.layer_output_size = layer_output_size
        self.freeze_all_layers = freeze_all_layers
        self.layer1 = nn.Linear(self.input_size, self.layer_output_size)
        self.layer2 = nn.Linear(self.layer_output_size, 1)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.calibration_layer = nn.Linear(self.input_size, self.input_size)
        self.calibration_layer.bias.data.zero_()
        self.calibration_layer.weight.data.fill_(1.0)
        self.calibration_layer.bias.requires_grad = False

        if pretrained_network is not None:
            self.layer1.load_state_dict(pretrained_network.layer1.state_dict())
            self.layer2.load_state_dict(pretrained_network.layer2.state_dict())
            if self.freeze_all_layers:
                freeze_layers = [self.layer1.parameters(), self.layer2.parameters()]
            else:
                freeze_layers = [self.layer1.parameters()]

            for param in freeze_layers:
                for p in param:
                    p.requires_grad = False

    def forward_layer(self, layer, x):
        """Apply one linear layer followed by Mish and dropout."""
        x = layer(x)
        x = F.mish(x)
        x = self.dropout(x)
        return x

    def get_weights(self):
        """Return the weight matrix of every linear layer as NumPy arrays."""
        weights = {}
        weights['calibration_layer'] = self.calibration_layer.detach().numpy()
        weights['layer1'] = self.layer1.weight.detach().numpy()
        weights['layer2'] = self.layer2.weight.detach().numpy()
        return weights

    def get_x_after_first_layer(self, x):
        """Representation after the calibration layer."""
        return self.forward_layer(self.calibration_layer, x)

    def get_x_after_second_layer(self, x):
        """Representation after the first hidden block."""
        x_after_first_layer = self.get_x_after_first_layer(x)
        return self.forward_layer(self.layer1, x_after_first_layer)

    def forward(self, x):
        """Return the raw logit for each row of ``x``."""
        x_after_second_layer = self.get_x_after_second_layer(x)
        return self.layer2(x_after_second_layer)


MODEL_SIZES = {
    "large": FNNClassifierTri3,
    "medium": FNNClassifierTri3_Medium,
    "small": FNNClassifierTri3_Small,
    "test": FNNClassifierTri3_Test,
    "calibration": FNNClassifierTri3_Calibration,
}
