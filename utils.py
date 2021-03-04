import os
import torch
from torchvision import transforms
from PIL import Image


def accuracy(outputs, target_label):
    total = 0.
    corrects = 0.
    for idx in range(len(outputs)):
        # output = torch.flatten(torch.sigmoid(outputs[idx]))
        output = torch.flatten((outputs[idx]))
        # out_labels = torch.zeros_like(output)ot
        # out_labels[output > 0.5] = 1
        pred = output > 0
        corrects += torch.sum(pred.view(-1) == target_label).float()
        total += output.view(-1).size(0)
    return corrects/total


def count_parameters(model, learnable_params_only=False):
    if learnable_params_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())
    pass


def export_image(images, output_path):
    if not isinstance(images, list):
        images = [images]
    height = 0
    width = 0
    for image in images:
        height = max(image.size(-2), height)
        width += image.size(-1)

    for idx, image in enumerate(images):
        images[idx] = (image + 1.) / 2.

    transf = transforms.ToPILImage()
    images = [transf(image.squeeze().cpu()) for image in images]
    image_out = Image.new(mode='RGB', size=(width, height))
    w = 0
    for idx, image in enumerate(images):
        h = (height - image.size[1]) // 2
        image_out.paste(image, (w, h))
        w += image.size[0]
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    image_out.save(output_path)


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if p.grad is not None:
                ave_grads.append(p.grad.abs().mean())
                print(n, p.grad.abs().mean())
