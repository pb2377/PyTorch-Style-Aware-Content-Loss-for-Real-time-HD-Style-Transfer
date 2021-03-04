import os
import torch
import models
import utils
import losses
import datasets
from torch.utils.data import DataLoader
from time import process_time


def main():
    train = True
    input_size = 768
    # crop_size = 256  # set to none for default cropping
    dual_optim = False
    print("Training with Places365 Dataset")
    max_its = 300000
    max_eps = 10000
    optimizer = 'adam'  # separate optimizers for discriminator and autoencoder
    lr = 0.01
    batch_size = 1
    step_lr_gamma = 0.1
    step_lr_step = 200000
    discr_success_rate = 0.8
    win_rate = 0.8
    log_interval = int(max_its // 20)
    log_interval = 2
    if log_interval < 10:
       print("\n WARNING: VERY SMALL LOG INTERVAL\n")

    lam = 0.001
    disc_wt = 1.
    trans_wt = 100.
    style_wt = 100.

    alpha = 0.05

    tblock_kernel = 11 if input_size == 768 else round(11 * input_size/768)
    tblock_kernel = max(tblock_kernel, 3)
    # Models
    encoder = models.Encoder()
    decoder = models.Decoder()
    tblock = models.TransformerBlock(kernel_size=tblock_kernel)
    discrim = models.Discriminator()
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        tblock = tblock.cuda()
        discrim = discrim.cuda()

    if train:
        # Losses
        gen_loss = losses.GeneratorLoss()
        disc_loss = losses.DiscriminatorLoss()
        transf_loss = losses.TransformedLoss()
        style_aware_loss = losses.StyleAwareContentLoss()

        # optimizer for encoder/decoder (and tblock? - think it has no parameters though)
        params_to_update = []
        for m in [encoder, decoder, tblock, discrim]:
            for param in m.parameters():
                param.requires_grad = True
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=lr)

        data_dir = '../Datasets/WikiArt-Sorted/data/vincent-van-gogh_road-with-cypresses-1890'
        style_data = datasets.StyleDataset(data_dir)
        num_workers = 8
        # if mpii:
        #     dataloaders = {'train': DataLoader(datasets.MpiiDataset(train=True, input_size=input_size,
        #                                                             style_dataset=style_data, crop_size=crop_size),
        #                                        batch_size=batch_size, shuffle=True, num_workers=num_workers),
        #                    'test': DataLoader(datasets.MpiiDataset(train=False, style_dataset=style_data, input_size=input_size),
        #                                       batch_size=1, shuffle=False, num_workers=num_workers)}
        # else:
        dataloaders = {'train': DataLoader(datasets.PlacesDataset(train=True, input_size=input_size,
                                                                  style_dataset=style_data),
                                           batch_size=batch_size, shuffle=True, num_workers=num_workers),
                       'test': DataLoader(datasets.TestDataset(),
                                          batch_size=1, shuffle=False, num_workers=num_workers)}

        # optimizer for encoder/decoder (and tblock? - think it has no parameters though)
        gen_params = []
        for m in [encoder, decoder, tblock]:
            for param in m.parameters():
                param.requires_grad = True
                gen_params.append(param)
        g_optimizer = torch.optim.Adam(gen_params, lr=lr)

        # optimizer for disciminator
        disc_params = []
        for param in discrim.parameters():
            param.requires_grad = True
            disc_params.append(param)
        d_optimizer = torch.optim.Adam(disc_params, lr=lr)

        scheduler_g = torch.optim.lr_scheduler.StepLR(g_optimizer, step_lr_step, gamma=step_lr_gamma, last_epoch=-1)
        scheduler_d = torch.optim.lr_scheduler.StepLR(d_optimizer, step_lr_step, gamma=step_lr_gamma, last_epoch=-1)

        its = 0
        print('Begin Training:')
        g_steps = 0
        d_steps = 0
        image_id = 0
        time_per_it = []
        if max_its is None:
            max_its = len(dataloaders['train'])

        # set models to train()
        encoder.train()
        decoder.train()
        tblock.train()
        discrim.train()
        d_loss = 0
        g_loss = 0
        for epoch in range(max_eps):
                if its > max_its:
                    break
                for images, style_images in dataloaders['train']:
                    t0 = process_time()
                    # utils.export_image(images[0, :, :, :], style_images[0, :, :, :], 'input_images.jpg')

                    # zero gradients
                    g_optimizer.zero_grad()
                    d_optimizer.zero_grad()

                    if its > max_its:
                        break

                    if torch.cuda.is_available():
                        images = images.cuda()
                        if style_images is not None:
                            style_images = style_images.cuda()

                    # autoencoder
                    emb = encoder(images)
                    stylized_im = decoder(emb)

                    # if training do losses etc
                    stylized_emb = encoder(stylized_im)
                    # add losses

                    # tblock
                    transformed_inputs, transformed_outputs = tblock(images, stylized_im)
                    # add loss

                    # discriminator
                    d_out_fake = discrim(stylized_im)  # keep attached to generator because grads needed
                    d_out_real_ph = discrim(images)
                    d_out_real_style = discrim(style_images)

                    # accuracy given all the images
                    d_acc_neg = utils.accuracy(d_out_real_ph, target_label=0) + utils.accuracy(d_out_fake,
                                                                                               target_label=0)
                    d_acc_pos = utils.accuracy(d_out_real_style, target_label=1)
                    d_acc = (d_acc_neg + d_acc_pos) / 3
                    d_acc_neg /= 2
                    gen_acc = utils.accuracy(d_out_fake, target_label=1)  # accuracy given only the output image

                    if discr_success_rate < win_rate:
                        # discriminator train step
                        d_out_fake = discrim(stylized_im.clone().detach())
                        # detach from generator, so not propagating unnecessary gradients

                        for idx in range(len(d_out_real_ph)):
                            inputs = [d_out_real_ph[idx], d_out_fake[idx], d_out_real_style[idx]]
                            targets = [0, 0, 1]
                            d_loss += disc_loss(inputs, targets)
                        d_loss *= disc_wt

                        d_loss.backward()
                        optimizer.step()
                        discr_success_rate = discr_success_rate * (1. - alpha) + alpha * d_acc
                        d_steps += 1
                    else:
                        # generator train step
                        # Generator
                        g_loss = disc_wt * gen_loss(d_out_fake, 1)
                        g_loss += trans_wt * transf_loss(transformed_inputs, transformed_outputs)
                        g_loss += style_wt * style_aware_loss(emb, stylized_emb)
                        g_loss.backward()
                        optimizer.step()
                        discr_success_rate = discr_success_rate * (1. - alpha) + alpha * (1. - gen_acc)
                        g_steps += 1

                    # print(g_loss.item(), g_steps, d_loss.item(), d_steps)
                    t1 = process_time()
                    time_per_it.append((t1-t0)/3600)
                    if len(time_per_it) > 100:
                        time_per_it.pop(0)

                    if not its % log_interval:
                        running_mean_it_time = sum(time_per_it)/len(time_per_it)
                        time_rem = (max_its - its + 1) * running_mean_it_time
                        print("{}/{} -- {} G Steps -- G Loss {:.2f} -- G Acc {:.2f} -"
                              "- {} D Steps -- D Loss {:.2f} -- D Acc {:.2f} -"
                              "- {:.2f} D Success -- {:.1f} Hours remaing...".format(its, max_its, g_steps,
                                                                                         g_loss,  gen_acc,
                                                                                         d_steps, d_loss, d_acc,
                                                                                         discr_success_rate, time_rem))

                        for idx in range(images.size(0)):
                            output_path = 'outputs/training/'.format(epoch)
                            if not os.path.exists(output_path):
                                os.makedirs(output_path)

                            output_path += 'iteration_{:06d}_example_{}.jpg'.format(its, idx)
                            utils.export_image([images[idx, :, :, :], style_images[idx, :, :, :], stylized_im[idx, :, :, :]], output_path)

                    its += 1
                    scheduler_g.step()
                    scheduler_d.step()

                    if not its % 10000:
                        if not os.path.exists('tmp'):
                            os.mkdir('tmp')
                        torch.save(encoder, "tmp/encoder.pt")
                        torch.save(decoder, "tmp/decoder.pt")
                        torch.save(tblock, "tmp/tblock.pt")
                        torch.save(discrim, "tmp/discriminator.pt")

        # only save if running on gpu (otherwise I'm just fixing bugs)
        torch.save(encoder, "encoder.pt")
        torch.save(decoder, "decoder.pt")
        torch.save(tblock, "tblock.pt")
        torch.save(discrim, "discriminator.pt")

        evaluate(encoder, decoder, dataloaders['test'])
    else:
        encoder = torch.load('encoder.pt', map_location='cpu')
        decoder = torch.load('decoder.pt', map_location='cpu')

        # encoder.load_state_dict(encoder_dict)
        # decoder.load_state_dict(decoder_dict)

        dataloader = DataLoader(datasets.TestDataset(),
                                batch_size=1, shuffle=False, num_workers=8)
        evaluate(encoder, decoder, dataloader)
        raise NotImplementedError('Not implemented standalone ')


def evaluate(encoder, decoder, dataloader):
    encoder.eval()
    decoder.eval()
    image_id = 0
    for images in dataloader:
        # raise NotImplementedError
        if torch.cuda.is_available():
            images = images.cuda()
            # if style_images is not None:
            #     style_images = style_images.cuda()

        # autoencoder
        emb = encoder(images)
        stylized_im = decoder(emb)

        # save out 5 example images
        print('Image 1')
        for idx in range(images.size(0)):
            output_path = 'final_outputs/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            output_path += 'Example_{}.jpg'.format(image_id)
            image_id += 1
            utils.export_image([images[idx, :, :, :], stylized_im[idx, :, :, :]],
                               output_path)
            # raise NotImplementedError("Not implemented test phase.")


if __name__ == '__main__':
    main()
