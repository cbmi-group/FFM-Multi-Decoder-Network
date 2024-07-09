from __future__ import print_function

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys
import shutil
import numpy as np
import time
import torch
import torch.nn as nn
import torchvision

root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "datasets"))
sys.path.append(os.path.join(root_dir, "models"))

from datasets.dataset import build_data_loader
from models.hrnet import HRNetV2
from models.optimize import create_criterion, create_optimizer, update_learning_rate, warmup_learning_rate
from datasets.metric import *
from hausdorff import hausdorff_distance

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


def print_table(data):
    col_width = [max(len(item) for item in col) for col in data]
    for row_idx in range(len(data[0])):
        for col_idx, col in enumerate(data):
            item = col[row_idx]
            align = '<' if not col_idx == 0 else '>'
            print(('{:' + align + str(col_width[col_idx]) + '}').format(item), end=" ")
        print()


def train_one_epoch(epoch, total_steps, dataloader, model,
                    device, criterion, optimizer, lr,
                    display_iter, log_file, warmup_step, warmup_method):
    model.train()
    smooth_loss = 0.0
    current_step = 0
    t0 = 0.0

    for inputs in dataloader:
        t1 = time.time()

        images = inputs['image'].to(device)
        labels = inputs['mask'].to(device)

        # forward pass
        seg = model(images)

        # compute loss
        loss = criterion(seg[0], labels)

        # predictions
        t0 += (time.time() - t1)

        total_steps += 1
        current_step += 1
        smooth_loss += loss.item()

        # backpropagate when training
        optimizer.zero_grad()
        # lr_update = update_learning_rate(optimizer, epoch, lr, step=lr_decay)
        lr_update = warmup_learning_rate(optimizer, total_steps, warmup_step, lr, warmup_method)
        loss.backward()
        optimizer.step()

        # torch.cuda.empty_cache()

        # log_loss
        if total_steps % display_iter == 0:
            smooth_loss = smooth_loss / current_step
            message = "Epoch: %d Step: %d LR: %.6f Loss: %.4f Runtime: %.2fs/%diters." % (
                epoch + 1, total_steps, lr_update, smooth_loss, t0, display_iter)
            print("==> %s" % (message))
            with open(log_file, "a+") as fid:
                fid.write('%s\n' % message)
        t0 = 0.0
        current_step = 0
        smooth_loss = 0.0
    return total_steps


def eval_one_epoch(epoch, dataloader, model, device, log_file, threshold):
    with torch.no_grad():
        model.eval()

        total_iou = 0.0
        total_f1 = 0.0
        total_acc = 0.0
        total_img = 0

        for inputs in dataloader:
            images = inputs['image'].to(device)
            labels = inputs['mask']

            total_img += len(images)

            outputs_list = model(images)
            outputs = outputs_list[0]

            preds = outputs > threshold
            preds = preds.cpu()

            val_acc = acc(preds, labels)
            total_acc += val_acc

            val_iou = IoU(preds, labels)
            total_iou += val_iou

            val_f1 = F1_score(preds, labels)
            total_f1 += val_f1
        # iou
        epoch_iou = total_iou / total_img
        epoch_f1 = total_f1 / total_img
        epoch_acc = total_acc / total_img

        message = "total Threshold: {:.3f} =====> Evaluation IOU: {:.4f}; F1_score: {:.4f}; ACC: {:.4f}".format(
            threshold, epoch_iou, epoch_f1, epoch_acc)
        print("==> %s" % (message))

        with open(log_file, "a+") as fid:
            fid.write('%s\n' % message)

        # torch.cuda.empty_cache()
    return epoch_acc, epoch_iou, epoch_f1


def train_eval_model(opts):
    num_epochs = opts["num_epochs"]
    train_batch_size = opts["train_batch_size"]
    val_batch_size = opts["eval_batch_size"]
    dataset_type = opts['dataset_type']
    dataset_name = opts['dataset_name']

    opti_mode = opts["optimizer"]
    loss_criterion = opts["loss_criterion"]
    lr = opts["lr"]
    wd = opts["weight_decay"]
    warmup_step = opts["warmup_step"]
    warmup_method = opts["warmup_method"]

    gpus = opts["gpu_list"].split(',')
    os.environ['CUDA_VISIBLE_DEVICE'] = opts["gpu_list"]
    train_dir = opts["log_dir"]
    fractal_dir = opts["fractal_dir"]

    train_data_dir = opts["train_data_dir"]
    eval_data_dir = opts["eval_data_dir"]

    pretrained = opts["pretrained_model"]
    resume = opts["resume"]
    display_iter = opts["display_iter"]
    threshold = opts["threshold"]

    save_epoch = opts["save_every_epoch"]

    log_file = os.path.join(train_dir, "log_file.txt")
    os.makedirs(train_dir, exist_ok=True)
    model_dir = os.path.join(train_dir, "code_backup")
    os.makedirs(model_dir, exist_ok=True)
    if resume is None and os.path.exists(log_file): os.remove(log_file)
    shutil.copy("./models/hrnet.py", os.path.join(model_dir, "hrnet.py"))
    shutil.copy("./train_hrnet.py", os.path.join(model_dir, "train_hrnet.py"))
    shutil.copy("./datasets/dataset.py", os.path.join(model_dir, "dataset.py"))

    ckt_dir = os.path.join(train_dir, "checkpoints")
    os.makedirs(ckt_dir, exist_ok=True)

    # format printing configs
    print("*" * 50)
    table_key = []
    table_value = []
    n = 0
    for key, value in opts.items():
        table_key.append(key)
        table_value.append(str(value))
        n += 1
    print_table([table_key, ["="] * n, table_value])

    # format gpu list
    gpu_list = []
    for str_id in gpus:
        id = int(str_id)
        gpu_list.append(id)

    # dataloader
    print("==> Create dataloader")
    dataloaders_dict = {
        "train": build_data_loader(dataset_name, train_data_dir, train_batch_size, dataset_type, is_train=True,
                                   fractal_dir=fractal_dir),
        "eval": build_data_loader(dataset_name, eval_data_dir, val_batch_size, dataset_type, is_train=False,
                                  fractal_dir=fractal_dir)}

    # define parameters of two networks
    print("==> Create network")

    num_classes = 1

    if 'fractal' in opts["dataset_type"]:
        if 'road' in opts["dataset_type"]:
            num_channels = 5
        else:
            num_channels = 3
    elif 'copy' in opts["dataset_type"]:
        if 'road' in opts["dataset_type"]:
            num_channels = 6
        else:
            num_channels = 3
    elif 'road' in opts["dataset_type"]:
        num_channels = 3
    else:
        num_channels = 1

    model = HRNetV2(n_channels=num_channels, n_class=num_classes)

    # loss layer
    criterion = create_criterion(criterion=loss_criterion)

    best_acc = 0.0
    best_iou = 0.0
    start_epoch = 0

    # load pretrained model
    if pretrained is not None and os.path.isfile(pretrained):
        print("==> Train from model '{}'".format(pretrained))
        checkpoint_gan = torch.load(pretrained)
        model.load_state_dict(checkpoint_gan['model_state_dict'])
        print("==> Loaded checkpoint '{}')".format(pretrained))
        for param in model.parameters():
            param.requires_grad = False

    # resume training
    elif resume is not None and os.path.isfile(resume):
        print("==> Resume from checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if
                           k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
        print("==> Loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch'] + 1))

    # train from scratch
    else:
        print("==> Train from initial or random state.")

    # define mutiple-gpu mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.cuda()
    hrnet_model = nn.DataParallel(model)

    # print learnable parameters
    print("==> List learnable parameters")
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t{}, size {}".format(name, param.size()))
    params_to_update = [{'params': model.parameters()}]

    # define optimizer
    print("==> Create optimizer")
    optimizer = create_optimizer(params_to_update, opti_mode, lr=lr, momentum=0.9, wd=wd)
    if resume is not None and os.path.isfile(resume): optimizer.load_state_dict(checkpoint['optimizer'])

    # start training
    since = time.time()

    # Each epoch has a training and validation phase
    print("==> Start training")
    total_steps = 0

    for epoch in range(start_epoch, num_epochs):

        print('-' * 50)
        print("==> Epoch {}/{}".format(epoch + 1, num_epochs))

        total_steps = train_one_epoch(epoch, total_steps,
                                      dataloaders_dict['train'],
                                      hrnet_model, device,
                                      criterion, optimizer, lr,
                                      display_iter, log_file, warmup_step, warmup_method)

        epoch_acc, epoch_iou, epoch_f1 = eval_one_epoch(epoch, dataloaders_dict['eval'], hrnet_model,
                                                        device, log_file, threshold)

        if best_iou < epoch_iou and epoch >= 3:
            best_iou = epoch_iou
            torch.save({'epoch': epoch,
                        'model_state_dict': hrnet_model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc': best_acc},
                       os.path.join(ckt_dir, "best.pth"))

        if (epoch + 1) % save_epoch == 0 and (epoch + 1) >= 25:
            torch.save({'epoch': epoch,
                        'model_state_dict': hrnet_model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc': epoch_acc,
                        'best_f1': epoch_f1,
                        'best_iou': epoch_iou},
                       os.path.join(ckt_dir, "checkpoints_" + str(epoch + 1) + ".pth"))
    time_elapsed = time.time() - since
    time_message = 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
    print(time_message)
    with open(log_file, "a+") as fid:
        fid.write('%s\n' % time_message)
    print('==> Best val Iou: {:4f}'.format(best_iou))
    with open(log_file, "a+") as fid:
        fid.write('==> Best val Iou: {:4f}'.format(best_iou))


if __name__ == '__main__':
    dataset_name = ['ER', 'MITO', 'ROSE', 'STARE', 'ROAD', 'NUCLEUS']
    dataset_list = ['er', 'er_fractal']
    date = '20240312'

    opts = dict()
    opts['dataset_type'] = 'er_fractal'
    opts["dataset_name"] = 'ER'
    opts["model_type"] = 'HRNet'
    opts["fractal_dir"] = 'Fractal_info_5'
    opts["num_epochs"] = 50
    opts["train_data_dir"] = "./dataset_txts/train_er.txt"
    opts["eval_data_dir"] = "./dataset_txts/test_er.txt"
    opts["train_batch_size"] = 32
    opts["eval_batch_size"] = 32
    opts["optimizer"] = "SGD"
    opts["loss_criterion"] = "iou"
    opts["threshold"] = 0.3
    opts["lr"] = 0.05
    opts["warmup_step"] = 1000
    opts["warmup_method"] = 'exp'
    opts["weight_decay"] = 0.0005
    opts["gpu_list"] = "0,1,2,3"

    log_dir = "./train_logs/" + str(opts["dataset_type"]) + "_" + opts["model_type"] + "_" + \
              opts["loss_criterion"] + "_" + str(opts["train_batch_size"]) + '_' + str(opts["lr"]) + \
              '_' + str(opts["num_epochs"]) + '_' + str(opts["threshold"]) + '_' + str(
        opts["warmup_step"]) + '_' + date + '_warmup_' + opts["fractal_dir"]

    opts["log_dir"] = log_dir
    opts["pretrained_model"] = None
    opts["resume"] = None
    opts["display_iter"] = 10
    opts["save_every_epoch"] = 5

    train_eval_model(opts)
