import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html
from torchsummary import summary

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.learn_residual = True
opt.dataset_mode = 'single'
opt.model = 'test'
opt.dataroot = 'datasets/summer2winter_yosemite/testA'
opt.checkpoints_dir = './checkpoints/'
opt.name = 'yosemite_cyclegan'
opt.which_model_netG = 'resnet_6blocks'
opt.ngf = 32
opt.loadSize = 256+15
opt.fineSize = 256
model = create_model(opt)
summary(model.netG, (3,256,256))