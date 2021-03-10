import torch
import torch.nn as nn


# class DiscriminatorLoss(nn.Module):
#     def __init__(self):
#         super(DiscriminatorLoss, self).__init__()
#         # pos_weight = torch.Tensor([2.])
#         # if torch.cuda.is_available():
#         #     pos_weight = pos_weight.cuda()
#         self.loss = nn.BCEWithLogitsLoss()
#
#     def forward(self, x, target_labels):
#         # log D(y)
#         # predict ones for real art, zeros for fake art and photos
#         # predict ones for
#         # reshape x, labels into vectors
#         x = [i.view(-1, 1) for i in x]
#         labels = torch.cat(tuple([torch.ones_like(x[0]) * i for i in target_labels]), dim=1)
#         x = torch.cat(tuple(x), dim=1)
#         loss = self.loss(x, labels)
#         return loss

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


# class DiscriminatorLoss(nn.Module):
#     def __init__(self):
#         super(DiscriminatorLoss, self).__init__()
#         self.loss = nn.BCEWithLogitsLoss()
#
#     def forward(self, x, target_label):
#         # log D(y)
#         # predict ones for real art, zeros for fake art and photos
#         # predict ones for
#         # reshape x, labels into vectors
#         loss = 0.
#         for idx in range(len(x)):
#             xi = x[idx].view(-1, 1)
#             labels = torch.ones_like(xi) * target_label
#             loss = self.loss(xi, labels)
#         return loss


# class GeneratorLoss(nn.Module):
#     def __init__(self):
#         super(GeneratorLoss, self).__init__()
#         self.loss = nn.BCEWithLogitsLoss()
#
#     def forward(self, x, target_label):
#         # log(1 - D(G(x)))
#         # predict ones for output images
#         # reshape x
#         loss = 0.
#         for idx in range(len(x)):
#             xi = x[idx].view(-1, 1)
#             labels = torch.ones_like(xi) * target_label
#             loss += self.loss(xi, labels)
#         return loss


class StyleAwareContentLoss(nn.Module):
    def __init__(self):
        super(StyleAwareContentLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, x_inputs, gx_outputs):
        # abs euclidean distance between photo embedding and and stylized emb
        return self.loss(x_inputs, gx_outputs)


class TransformedLoss(nn.Module):
    def __init__(self):
        super(TransformedLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, x_inputs, gx_outputs):
        return self.loss(x_inputs, gx_outputs)
