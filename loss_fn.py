import torch

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # Apply sigmoid activation to convert logits to probabilities
        inputs = inputs.view(inputs.size(0),inputs.size(1),-1)
        targets = targets.view(targets.size(0),targets.size(1),-1)

        intersection = (inputs * targets).sum()
        dice_coefficient = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice_coefficient
