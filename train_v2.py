"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import os

import matplotlib.pyplot as plt
from copy import deepcopy
from tifffile import imsave


def print_current_losses(epoch, iters, losses, t_comp, t_data, log_name, phase='train'):
    """print current losses on console; also save the losses to the disk

    Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        t_comp (float) -- computational time per data point (normalized by batch_size)
        t_data (float) -- data loading time per data point (normalized by batch_size)
        log_name (str) -- log_file
    """
    message = '(epoch: %d, iters: %d, time: %.6f, data: %.6f, phase: %s) ' % (epoch, iters, t_comp, t_data, phase)
    for k, v in losses.items():
        message += '%s: %.6f ' % (k, v)

    print(message)  # print the message
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    # opt.max_dataset_size = 1000
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training image batches = %d' % dataset_size)

    #### validation options
    val_opt = deepcopy(opt)
    val_opt.phase = 'valid'
    val_opt.serial_batches = True
    val_opt.max_dataset_size = 10 * opt.batch_size
    val_dataset = create_dataset(val_opt)  # create a dataset given opt.dataset_mode and other options
    val_dataset_size = len(val_dataset)    # get the number of images in the dataset.
    print('The number of validation image batches = %d' % val_dataset_size)

    #### validation options end

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)

    figures_save_path = '{}/{}/figures'.format(opt.checkpoints_dir, opt.name)
    os.makedirs(figures_save_path, exist_ok=True)

    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.train()                   # set model to train mode!
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += 1#opt.batch_size
            epoch_iter += 1#opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                print_current_losses(epoch, epoch_iter, losses, t_comp, t_data, log_name)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            # model.save_networks('latest')
            model.save_networks(epoch)

        #### Evaluate
        model.eval()                                      # set model to eval mode!
        val_iter_data_time = time.time()
        for i, data in enumerate(val_dataset):
            val_iter_start_time = time.time()
            val_t_data = val_iter_start_time - val_iter_data_time
            model.set_input(data)
            model.validate()
            losses = model.get_current_losses()
            val_t_comp = (time.time() - val_iter_start_time)
            print_current_losses(epoch, i+1, losses, val_t_comp, val_t_data, log_name, 'val')

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
