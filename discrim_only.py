import os
import torch
import models
import utils
import losses
import datasets
from torch.utils.data import DataLoader
from time import process_time


def main():
    input_size = 256
    dual_optim = False
    mpii = False
    print("Training with {} Dataset".format('MPII Human Pose' if mpii else 'Places365'))
    # input_size = 256
    max_its = 300000
    max_eps = 100
    optimizer = 'adam'  # separate optimizers for discriminator and autoencoder
    lr = 0.0002
    batch_size = 1
    step_lr_gamma = 0.1
    step_lr_step = 200000
    discr_success_rate = 0.8
    log_interval = int(max_its // 20)
    #
    lam = 0.001
    alpha = 0.05

    tblock_kernel = 11 if input_size == 768 else round(11 * input_size/768)
    tblock_kernel = max(tblock_kernel, 3)
    # Models
    encoder = models.Encoder()
    decoder = models.Decoder()
    tblock = models.TransformerBlock(kernel_size=tblock_kernel)
    discrim = models.Discriminator(input_size=input_size)
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        tblock = tblock.cuda()
        discrim = discrim.cuda()

    # Losses
    gen_loss = losses.GeneratorLoss()
    disc_loss = losses.DiscriminatorLoss()
    transf_loss = losses.TransformedLoss()
    style_aware_loss = losses.StyleAwareContentLoss()

    # optimizer for encoder/decoder (and tblock? - think it has no parameters though)
    gen_params = []
    for m in [encoder, decoder, tblock]:
        for param in m.parameters():
            param.requires_grad = True
            gen_params.append(param)
    optimizer_gen = torch.optim.Adam(gen_params, lr=lr)

    # glob_path = '../Datasets/BAM/BAM-Images/oilpx/aint/*/*'
    style_data = datasets.StyleDataset()
    num_workers = 8
    if mpii:
        dataloaders = {'train': DataLoader(datasets.MpiiDataset(train=True, input_size=input_size,
                                                                style_dataset=style_data),
                                           batch_size=batch_size, shuffle=True, num_workers=num_workers),
                       'test': DataLoader(datasets.MpiiDataset(train=False, style_dataset=style_data, input_size=input_size),
                                          batch_size=1, shuffle=False, num_workers=num_workers)}
    else:
        dataloaders = {'train': DataLoader(datasets.PlacesDataset(train=True, input_size=input_size,
                                                                  style_dataset=style_data),
                                           batch_size=batch_size, shuffle=True, num_workers=num_workers),
                       'test': DataLoader(datasets.TestDataset(style_dataset=style_data, input_size=input_size),
                                          batch_size=1, shuffle=False, num_workers=num_workers)}

    # optimizer for disciminator
    disc_params = []
    for param in discrim.parameters():
        param.requires_grad = True
        disc_params.append(param)
    optimizer_disc = torch.optim.Adam(disc_params, lr=lr) #, weight_decay=1e-4)
    # optimizer_disc = torch.optim.SGD(disc_params, lr=lr, momentum=0.9)

    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_gen, step_lr_step, gamma=step_lr_gamma, last_epoch=-1)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_disc, step_lr_step, gamma=step_lr_gamma, last_epoch=-1)

    its = 0
    print('Begin Training:')
    g_steps = 0
    d_steps = 0
    image_id = 0
    time_per_it = []
    if max_its is None:
        max_its = len(dataloaders['train'])
    for epochs in range(max_eps):
        for phase in ['train', 'test']:
            if its > max_its:
                break
            if phase == 'train':
                # set models to train()
                encoder.train()
                decoder.train()
                tblock.train()
                discrim.train()
            else:
                # set models to eval()
                encoder.eval()
                decoder.eval()
                tblock.eval()
                discrim.eval()
                image_id = 0

            for images, style_images in dataloaders[phase]:
                t0 = process_time()
                # utils.export_image(images[0, :, :, :], style_images[0, :, :, :], 'input_images.jpg')
                if its > max_its:
                    break
                # raise NotImplementedError
                if torch.cuda.is_available():
                    images = images.cuda()
                    if style_images is not None:
                        style_images = style_images.cuda()

                # # autoencoder
                # emb = encoder(images)
                # stylized_im = decoder(emb)

                # if training do losses etc
                if phase == 'train':
                    # stylized_emb = encoder(stylized_im)
                    # add losses

                    # tblock
                    # transformed_inputs, transformed_outputs = tblock(images, stylized_im)
                    # add loss

                    # Generator
                    # optimizer_gen.zero_grad()
                    # d_out_fake = discrim(stylized_im)
                    # g_loss = lam * gen_loss(d_out_fake, target_label=1)
                    # g_loss += transf_loss(transformed_inputs, transformed_outputs)
                    # g_loss += style_aware_loss(emb, stylized_emb)

                    # if dual_optim:
                    #     g_loss.backward()
                    #     optimizer_gen.step()

                    # discriminator
                    optimizer_disc.zero_grad()
                    d_out_real_ph = discrim(images)
                    # d_out_fake = discrim(stylized_im.detach())
                    d_out_real_style = discrim(style_images)

                    d_loss = 0.
                    for idx in range(len(d_out_real_ph)):
                        inputs = [d_out_real_ph[idx], d_out_real_style[idx]]
                        targets = [0, 1]
                        d_loss += disc_loss(inputs, targets)
                    # print(d_loss.item())
                    # # raise NotImplementedError
                    # # d_loss = disc_loss()
                    # d_loss = disc_loss(d_out_real_ph, target_label=0)  # real photo = 0
                    # # d_loss += disc_loss(d_out_fake, target_label=0)  # fake art = 0
                    # d_loss += disc_loss(d_out_real_style, target_label=1)  # real style = 1
                    # # d_loss *= lam
                    # print(d_loss.item())
                    # raise NotImplementedError
                    # # if dual_optim:
                    # #     d_loss.backward()
                    #     optimizer_disc.step()

                    # accuracy given all the images
                    d_acc_neg = utils.accuracy(d_out_real_ph, target_label=0)
                    d_acc_pos = utils.accuracy(d_out_real_style, target_label=1)
                    d_acc = (d_acc_neg + d_acc_pos) / 2
                    # gen_acc = utils.accuracy(d_out_fake, target_label=1)  # accuracy given only the output image

                    print("Loss {:.2f} -- Acc {:.2f} -"
                          "- Pos Acc {:.2f} -- Neg Acc {:.2f}".format(d_loss.item(), d_acc,
                                                                       d_acc_pos.item(), d_acc_neg.item()))
#                    d_acc = 0.0

                    d_loss.backward()
                    # print('\nD STEP')
                    # utils.plot_grad_flow(discrim.named_parameters())
                    optimizer_disc.step()
                    discr_success_rate = discr_success_rate * (1. - alpha) + alpha * d_acc
                    d_steps += 1

                    # # # Select what to update
                    # if not dual_optim:
                    #     if d_acc < discr_success_rate:
                    #         d_loss.backward()
                    #         # print('\nD STEP')
                    #         # utils.plot_grad_flow(discrim.named_parameters())
                    #         optimizer_disc.step()
                    #         discr_success_rate = discr_success_rate * (1. - alpha) + alpha * d_acc
                    #         d_steps += 1
                    #     else:
                    #         g_loss.backward()
                    #         # print('\nG STEP')
                    #         # utils.plot_grad_flow(encoder.named_parameters())
                    #         # utils.plot_grad_flow(decoder.named_parameters())
                    #         # utils.plot_grad_flow(tblock.named_parameters())
                    #         optimizer_gen.step()
                    #         discr_success_rate = discr_success_rate * (1. - alpha) + alpha * (1. - gen_acc)
                    #         g_steps += 1
                    # print(g_loss.item(), g_steps, d_loss.item(), d_steps)
                    t1 = process_time()
                    time_per_it.append((t1-t0)/3600)
                    if len(time_per_it) > 100:
                        time_per_it.pop(0)

                    if not its % log_interval:
                        running_mean_it_time = sum(time_per_it)/len(time_per_it)
                        time_rem = (max_its - its + 1) * running_mean_it_time
                        print("{}/{} -- {} G Steps -- G Loss {:.4f} -- G Acc {:.2f} -"
                              "- {} D Steps -- D Loss {:.4f} -- D Acc {:.2f} -"
                              "- {:.1f} Hours Remaining".format(its, max_its, g_steps, 0.0,  0.0,
                                                                d_steps, d_loss, d_acc, time_rem))

                        for idx in range(images.size(0)):
                            output_path = 'outputs/training/'.format(epochs)
                            if not os.path.exists(output_path):
                                os.makedirs(output_path)

                            output_path += 'iteration_{:06d}_example_{}.jpg'.format(its, idx)
                            utils.export_image([images[idx, :, :, :], style_images[idx, :, :, :]], output_path)
                    its += 1
                    scheduler_d.step()
                    scheduler_g.step()
                else:
                    # save out 5 example images
                    for idx in range(images.size(0)):
                        output_path = 'outputs/epoch_{}/'.format(epochs)
                        if not os.path.exists(output_path):
                            os.makedirs(output_path)

                        output_path += 'Example_{}.jpg'.format(image_id)
                        image_id += 1
                        utils.export_image(images[idx, :, :, :], stylized_im[idx, :, :, :], output_path)
                        # raise NotImplementedError("Not implemented test phase.")
            if phase == 'train':
                epochs += 1

    # only save if running on gpu (otherwise I'm just fixing bugs)
    if torch.cuda.is_available():
        torch.save(encoder, "encoder.pt")
        torch.save(decoder, "decoder.pt")
        torch.save(tblock, "tblock.pt")
        torch.save(discrim, "discriminator.pt")

    evaluate(encoder, decoder, dataloaders['test'])


def evaluate(encoder, decoder, dataloader):
    encoder.eval()
    decoder.eval()
    image_id = 0
    for images, style_images in dataloader:
        # raise NotImplementedError
        if torch.cuda.is_available():
            images = images.cuda()
            if style_images is not None:
                style_images = style_images.cuda()

        # autoencoder
        emb = encoder(images)
        stylized_im = decoder(emb)

        # save out 5 example images
        output_path = 'final_outputs/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_path += 'Example_{}.jpg'.format(image_id)
        image_id += 1
        utils.export_image(images, stylized_im, output_path)
        # raise NotImplementedError("Not implemented test phase.")


if __name__ == '__main__':
    main()
