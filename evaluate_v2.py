"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import numpy as np
import time


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    # opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'evaluate_v2_log.txt')
    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Evaluate Log (%s) ================\n' % now)
        log_file.write('{}\n'.format(opt.phase))
    start_time = time.time()

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    model.eval()
    loss_l1 = 0
    loss_l2 = 0
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        current_l1, current_l2 = model.evaluate_v2()           # run inference
        loss_l1 += current_l1
        loss_l2 += current_l2
        if i%100==0:
            message = 'Epoch: {}, Batch: {}/{}'.format(opt.epoch, i, len(dataset))
            print(message)  # print the message
            with open(log_name, "a") as log_file:
                log_file.write('%s\n' % message)  # save the message
    loss_l1 = (loss_l1 / len(dataset))
    loss_l2 = (loss_l2 / len(dataset))
    message = 'L1 Loss: {}, L2 Loss: {}'.format(loss_l1, loss_l2)
    print(message)
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message

    message = 'Time taken: {}'.format(time.time() - start_time)
    print(message)
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message
