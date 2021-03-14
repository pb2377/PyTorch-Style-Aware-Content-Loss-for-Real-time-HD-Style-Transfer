import os
import glob
import torch
import models
import utils
import losses
import datasets
from torch.utils.data import DataLoader
from time import process_time
from itertools import cycle

artist_list = ['van-gogh', 'cezanne', 'picasso', 'guaguin', 'kandisky', 'monet']


def main():
    train = True
    input_size = 768
    artist = 'cezanne'
    assert artist in artist_list
    # input_size = 256  # set to none for default cropping
    dual_optim = False
    print("Training with Places365 Dataset")
    max_its = 300000
    max_eps = 20000
    optimizer = 'adam'  # separate optimizers for discriminator and autoencoder
    lr = 0.0002
    batch_size = 1
    step_lr_gamma = 0.1
    step_lr_step = 200000
    discr_success_rate = 0.8
    win_rate = 0.8
    log_interval = int(max_its // 100)
    # log_interval = 100
    if log_interval < 10:
       print("\n WARNING: VERY SMALL LOG INTERVAL\n")

    lam = 0.001
    disc_wt = 1.
    trans_wt = 100.
    style_wt = 100.

    alpha = 0.05

    tblock_kernel = 10
    # Models
    encoder = models.Encoder()
    decoder = models.Decoder()
    tblock = models.TransformerBlock(kernel_size=tblock_kernel)
    discrim = models.Discriminator()

    # init weights
    models.init_weights(encoder)
    models.init_weights(decoder)
    models.init_weights(tblock)
    models.init_weights(discrim)

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        tblock = tblock.cuda()
        discrim = discrim.cuda()

    artist_dir = glob.glob('../Datasets/WikiArt-Sorted/data/*')
    for item in artist_dir:
        if artist in os.path.basename(item):
            data_dir = item
            break
    print('Retrieving style examples from {} artwork from directory {}'.format(artist.upper(), data_dir))

    save_dir = 'outputs-{}'.format(artist)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print('Saving weights and outputs to {}'.format(save_dir))

    if train:
        # load tmp weights
        if os.path.exists('tmp'):
            print('Loading from tmp...')
            assert os.path.exists("tmp/encoder.pt")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            encoder = torch.load("tmp/encoder.pt", map_location=device)
            decoder = torch.load("tmp/decoder.pt", map_location=device)
            tblock = torch.load("tmp/tblock.pt", map_location=device)
            discrim = torch.load("tmp/discriminator.pt", map_location=device)

        # Losses
        gen_loss = losses.SoftmaxLoss()
        disc_loss = losses.SoftmaxLoss()
        transf_loss = losses.TransformedLoss()
        style_aware_loss = losses.StyleAwareContentLoss()

        # # optimizer for encoder/decoder (and tblock? - think it has no parameters though)
        # params_to_update = []
        # for m in [encoder, decoder, tblock, discrim]:
        #     for param in m.parameters():
        #         param.requires_grad = True
        #         params_to_update.append(param)
        # # optimizer = torch.optim.Adam(params_to_update, lr=lr)
        style_data = datasets.StyleDataset(data_dir)
        num_workers = 8
        # if mpii:
        #     dataloaders = {'train': DataLoader(datasets.MpiiDataset(train=True, input_size=input_size,
        #                                                             style_dataset=style_data, crop_size=crop_size),
        #                                        batch_size=batch_size, shuffle=True, num_workers=num_workers),
        #                    'test': DataLoader(datasets.MpiiDataset(train=False, style_dataset=style_data, input_size=input_size),
        #                                       batch_size=1, shuffle=False, num_workers=num_workers)}
        # else:
        dataloaders = {'train': DataLoader(datasets.PlacesDataset(train=True, input_size=input_size),
                                           batch_size=batch_size, shuffle=True, num_workers=num_workers),
                       'style': DataLoader(datasets.StyleDataset(data_dir=data_dir, input_size=input_size),
                                           batch_size=batch_size, shuffle=True, num_workers=num_workers),
                       'test': DataLoader(datasets.TestDataset(),
                                          batch_size=1, shuffle=False, num_workers=num_workers)}

        # optimizer for encoder/decoder (and tblock? - think it has no parameters though)
        gen_params = []
        for m in [encoder, decoder]:
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
        gen_acc = 0
        d_acc = 0
        for epoch in range(max_eps):
                if its > max_its:
                    break
                for images, style_images in zip(dataloaders['train'], cycle(dataloaders['style'])):
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

                    if discr_success_rate < win_rate:
                        # discriminator train step
                        # discriminator
                        # detach from generator, so not propagating unnecessary gradients
                        d_out_fake = discrim(stylized_im.clone().detach())
                        d_out_real_ph = discrim(images)
                        d_out_real_style = discrim(style_images)

                        # accuracy given all the images
                        d_acc_real_ph = utils.accuracy(d_out_real_ph, target_label=0)
                        d_acc_fake_style = utils.accuracy(d_out_fake, target_label=0)
                        d_acc_real_style = utils.accuracy(d_out_real_style, target_label=1)
                        gen_acc = 1 - d_acc_fake_style
                        d_acc = (d_acc_real_ph + d_acc_fake_style + d_acc_real_style) / 3

                        # Loss calculation
                        d_loss = disc_loss(d_out_fake, target_label=0)
                        d_loss += disc_loss(d_out_real_style, target_label=1)
                        d_loss += disc_loss(d_out_real_ph, target_label=0)

                        # Step optimizer
                        d_loss.backward()
                        d_optimizer.step()
                        d_steps += 1

                        # Update success rate
                        discr_success_rate = discr_success_rate * (1. - alpha) + alpha * d_acc
                    else:
                        # generator train step
                        # Generator discrim losses

                        # discriminator
                        d_out_fake = discrim(stylized_im)  # keep attached to generator because grads needed

                        # accuracy given the fake output, generator images
                        gen_acc = utils.accuracy(d_out_fake, target_label=1)  # accuracy given only the output image

                        del g_loss
                        # tblock
                        transformed_inputs, transformed_outputs = tblock(images, stylized_im)

                        g_loss = disc_wt * gen_loss(d_out_fake, target_label=1)
                        g_transf = trans_wt * transf_loss(transformed_inputs, transformed_outputs)
                        g_style = style_wt * style_aware_loss(emb, stylized_emb)
                        # print(g_loss.item(), g_transf.item(), g_style.item())
                        g_loss += g_transf + g_style

                        # STEP OPTIMIZER
                        g_loss.backward()
                        g_optimizer.step()
                        g_steps += 1

                        # Update success rate
                        discr_success_rate = discr_success_rate * (1. - alpha) + alpha * (1. - gen_acc)

                    # report stuff
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
                            output_path = os.path.join(save_dir, 'training_visualise')
                            if not os.path.exists(output_path):
                                os.makedirs(output_path)

                            output_path = os.path.join(output_path, 'iteration_{:06d}_example_{}.jpg'.format(its, idx))
                            utils.export_image([images[idx, :, :, :], style_images[idx, :, :, :], stylized_im[idx, :, :, :]], output_path)

                    its += 1
                    scheduler_g.step()
                    scheduler_d.step()

                    if not its % 10000:
                        if not os.path.exists(save_dir):
                            os.mkdir(save_dir)
                        torch.save(encoder, save_dir + "/encoder.pt")
                        torch.save(decoder, save_dir + "/decoder.pt")
                        torch.save(tblock, save_dir + "/tblock.pt")
                        torch.save(discrim, save_dir + "/discriminator.pt")

        # only save if running on gpu (otherwise I'm just fixing bugs)
        torch.save(encoder, os.path.join(save_dir, "encoder.pt"))
        torch.save(decoder, os.path.join(save_dir, "decoder.pt"))
        torch.save(tblock, os.path.join(save_dir, "tblock.pt"))
        torch.save(discrim, os.path.join(save_dir, "discriminator.pt"))

        evaluate(encoder, decoder, dataloaders['test'], save_dir=save_dir)
    else:
        print('Loading Models {} and {}'.format(os.path.join(save_dir, "encoder.pt"), os.path.join(save_dir, "decoder.pt")))
        encoder = torch.load(os.path.join(save_dir, "encoder.pt"), map_location='cpu')
        decoder = torch.load(os.path.join(save_dir, "decoder.pt"), map_location='cpu')

        # encoder.load_state_dict(encoder_dict)
        # decoder.load_state_dict(decoder_dict)
        if torch.cuda.is_available():
            encoder = encoder.cuda()
            decoder = decoder.cuda()

        dataloader = DataLoader(datasets.TestDataset(input_size=input_size),
                                batch_size=1, shuffle=False, num_workers=8)
        evaluate(encoder, decoder, dataloader, save_dir=save_dir)
        # raise NotImplementedError('Not implemented standalone ')


def evaluate(encoder, decoder, dataloader, save_dir):
    encoder.eval()
    decoder.eval()
    image_id = 0
    for images in dataloader:
        # if True:
        if image_id in [18, 20, 25]: #[12, 17, 25]:
            # raise NotImplementedError
            if torch.cuda.is_available():
                images = images.cuda()
                # if style_images is not None:
                #     style_images = style_images.cuda()

            # autoencoder
            emb = encoder(images)
            stylized_im = decoder(emb)

            # save out 5 example images
            for idx in range(images.size(0)):
                output_path = os.path.join(save_dir, 'final_outputs')
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                output_path = os.path.join(output_path, 'Example_{}.jpg'.format(image_id))
                utils.export_image([images[idx, :, :, :], stylized_im[idx, :, :, :]],
                                   output_path)
                # raise NotImplementedError("Not implemented test phase.")
                print('Image {}/{}'.format(image_id+1,  len(dataloader.dataset)+1))
        image_id += 1

if __name__ == '__main__':
    main()
