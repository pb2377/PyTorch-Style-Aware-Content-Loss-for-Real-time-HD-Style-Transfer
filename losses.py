import torch
import torch.nn as nn


class SoftmaxLoss(nn.Module):
    def __init__(self):
        super(SoftmaxLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, x, target_label):
        # loss D(y)
        # predict ones for real art, zeros for fake art and photos
        loss = 0.
        for output in x:
            labels = torch.ones_like(output) * target_label
            loss += self.loss(output, labels)
        return loss


class StyleAwareContentLoss(nn.Module):
    def __init__(self):
        super(StyleAwareContentLoss, self).__init__()
        self.loss = nn.L1Loss(reduction='mean')
        # self.loss = nn.MSELoss()

    def forward(self, x_inputs, gx_outputs):
        # abs euclidean distance between photo embedding and and stylized emb
        return self.loss(x_inputs, gx_outputs)


class TransformedLoss(nn.Module):
    def __init__(self):
        super(TransformedLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, x_inputs, gx_outputs):
        return self.loss(x_inputs, gx_outputs)
