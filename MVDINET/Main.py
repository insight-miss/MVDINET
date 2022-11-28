from __future__ import print_function
import argparse

import os
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from sklearn.metrics import accuracy_score, f1_score
import time
import DeterminDimension
from DeepFeatureBySVM import level4_class, level4_num_class
from MyDataset import MyDataset
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset

from github import Model

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./data/')
parser.add_argument('--mode', default='train', help='train|test')
parser.add_argument('--outf', default='./results/')
parser.add_argument('--net', default='', help='use the saved model')
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--batchSize', type=int, default=64, help='the mini-batch size of training')
parser.add_argument('--testSize', type=int, default=64)
parser.add_argument('--epochs', type=int, default=50, help='the number of epochs')
parser.add_argument('--num_classes', type=int, default=7, help='the number of classes')
parser.add_argument('--num_view', type=int, default=2, help='the number of views')
parser.add_argument('--fea_out', type=int, default=200, help='the dimension of the first linear layer')
parser.add_argument('--fea_com', type=int, default=300, help='the dimension of the combination layer')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--print_freq', '-p', default=1, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gamma', type=float, default=3.0, help='the power of the weight for each view')
parser.add_argument('--savepath', default='./save/')
opt = parser.parse_args()
opt.cuda = False
cudnn.benchmark = True


# ======================================= Define functions =============================================
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opt.lr * (0.05 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, weight_var, gamma, criterion, optimizer, epoch, F_txt):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for index, (sample_set, sample_targets) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input_var = [sample_set[i].type(torch.FloatTensor).cuda() for i in range(len(sample_set))]

        target_var = sample_targets.long().cuda()

        Output_list = model(input_var)

        weight_up_list = []
        loss = torch.zeros(1).cuda()

        for v in range(len(Output_list)):
            loss_temp = criterion(Output_list[v], target_var)
            loss += (weight_var[v] ** gamma) * loss_temp
            weight_up_temp = loss_temp ** (1 / (1 - gamma))
            weight_up_list.append(weight_up_temp)

        output_var = torch.stack(Output_list)

        weight_var = weight_var.unsqueeze(1)
        weight_var = weight_var.unsqueeze(2)
        weight_var = weight_var.expand(weight_var.size(0), target_var.size(0), opt.num_classes)
        output_weighted = weight_var * output_var
        output_weighted = torch.sum(output_weighted, 0)

        weight_var = weight_var[:, :, 1]
        weight_var = weight_var[:, 1]

        # measure accuracy and record loss
        result, prec1 = accuracy(output_weighted, target_var)

        # print(target.size(0))
        losses.update(loss.item(), target_var.size(0))
        top1.update(prec1[0], target_var.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if index % opt.print_freq == 0:
            print('Train-Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, index, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

            print('Train-Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, index, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1), file=F_txt)

    return weight_var

def data_test(val_loader, model, weight_var, gamma, criterion, best_prec1, F_txt):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    pred_ec = []
    target_ec = []
    with torch.no_grad():

        end = time.time()

        for index, (sample_set, sample_targets) in enumerate(val_loader):

            input_var = [sample_set[i].type(torch.FloatTensor).cuda() for i in range(len(sample_set))]
            # deal with the target
            target_var = sample_targets.long().cuda()

            Output_list = model(input_var)
            loss = torch.zeros(1).cuda()

            for v in range(len(Output_list)):
                loss_temp = criterion(Output_list[v], target_var)

                loss += (weight_var[v] ** gamma) * loss_temp

            output_var = torch.stack(Output_list)
            weight_var = weight_var.unsqueeze(1)
            weight_var = weight_var.unsqueeze(2)
            weight_var = weight_var.expand(weight_var.size(0), target_var.size(0), opt.num_classes)
            output_weighted = weight_var * output_var
            output_weighted = torch.sum(output_weighted, 0)

            weight_var = weight_var[:, :, 1]
            weight_var = weight_var[:, 1]

            # measure accuracy and record loss
            result, prec1 = accuracy(output_weighted, target_var)
            losses.update(loss.item(), target_var.size(0))
            top1.update(prec1[0], target_var.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            for kk in range(result.shape[0]):
                pred_ec.append(result[kk].item())
                target_ec.append(target_var[kk].item())

            if index % opt.print_freq == 0:
                print('Test: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, index, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=top1))

                print('Test: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, index, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=top1), file=F_txt)

        print(' * Prec@1 {top1.avg:.3f} Best_Prec@1 {best:.3f}'.format(top1=top1,
                                                                       best=best_prec1))
        print(' * Prec@1 {top1.avg:.3f} Best_Prec@1 {best:.3f}'.format(top1=top1,
                                                                       best=best_prec1),
              file=F_txt)

    return top1.avg, pred_ec, target_ec


def save_checkpoint(state, is_best, foldPath, fold, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        file_model_best = os.path.join(foldPath, 'fold' + str(fold) + '.tar')
        shutil.copyfile(filename, file_model_best)

        os.remove(filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.view(target.shape)
        correct = pred.eq(target)

        correct = correct.view(-1).float().sum(0, keepdim=True)
        res = correct.mul_(100.0 / batch_size)
        return pred, res


def init_model(train_loader, num_class=7):
    train_iter = iter(train_loader)
    traindata, target = train_iter.next()

    view_list = []
    for v in range(len(traindata)):
        temp_size = traindata[v].size()
        view_list.append(temp_size[1])

    model = Model.define_MVDINET(use_gpu=opt.cuda, view_list=view_list,
                                             fea_out=opt.fea_out, fea_com=opt.fea_com)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))

    return model, criterion, optimizer


def fold_data_loader(train_data, test_data, batch_size=64, level=2, exclude=[1], ecs=5):
    train_data, train_label, _ = DeterminDimension.final_data(train_data, level=level, exclude=exclude, ecs=ecs)
    test_data, test_label, _ = DeterminDimension.final_data(test_data, level=level, exclude=exclude, ecs=ecs)

    print("train: %d, test:%d" % (len(train_label), len(test_label)))

    train_dataset = MyDataset(train_data, train_label)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True)

    test_dataset = MyDataset(test_data, test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

    return train_dataloader, test_dataloader


def get_mean_std(list):
    return np.mean(list), np.std(list, ddof=1)


feature_path = opt.data_path

num_classes = [2, 402]

for the_level in range(2):

    # current_level = the_level + 1
    current_level = the_level

    savePath = opt.outf + "Level" + str(current_level)
    opt.num_classes = num_classes[current_level]

    opt.fea_out = 512
    opt.fea_com = 512

    if not os.path.exists(savePath):
        os.makedirs(savePath)
    final_result_path = os.path.join(savePath, 'final_result.txt')
    final_T = open(final_result_path, 'a+')
    acc_list = []
    f1_list = []
    for k in range(5):

        exclude = [1]

        if 1 not in exclude:
            opt.num_view = 4

        if 3 not in exclude:
            opt.num_view = 3
        # save the opt and results to txt file
        flod = k + 1

        # save the opt and results to txt file
        foldPath = savePath + "/fold" + str(flod)
        if not os.path.exists(foldPath):
            os.makedirs(foldPath)

        txt_save_path = os.path.join(foldPath, 'opt_results.txt')
        F_txt = open(txt_save_path, 'a+')

        # ============================================ Loader data ========================================
        if num_classes[current_level] == 2:
            test_data = pd.read_csv(feature_path + "enzyme_no_enzyme_test" + str(flod) + ".csv")
            train_data = pd.read_csv(feature_path + "enzyme_no_enzyme_train" + str(flod) + ".csv")
        else:
            test_data = pd.read_csv(feature_path + "enzyme/test" + str(flod) + ".csv")
            train_data = pd.read_csv(feature_path + "enzyme/train" + str(flod) + ".csv")

        # ============================================ Loader model ========================================
        train_loader, test_loader = fold_data_loader(train_data, test_data, level=current_level, exclude=exclude)
        model, criterion, optimizer = init_model(train_loader, num_class=num_classes[current_level])
        # ============================================ Training phase ========================================
        print('start training.........')
        start_time = time.time()
        weight_var = torch.ones(opt.num_view * 2) * (1 / (opt.num_view * 2))
        gamma = torch.tensor(opt.gamma).cuda()
        weight_var = weight_var.cuda()

        best_prec = 0
        test_pre = []
        test_label = []
        for epoch in range(opt.epochs):
            # adjust the learning rate
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            weight_var = train(train_loader, model, weight_var, gamma, criterion, optimizer, epoch, F_txt)

            # evaluate on test
            prec, pre_ec, target_ec = data_test(test_loader, model, weight_var, gamma, criterion, best_prec, F_txt)

            # remember best prec and save checkpoint
            prec = accuracy_score(target_ec, pre_ec)
            is_best = prec > best_prec
            best_prec = max(prec, best_prec)

            "save best"
            if is_best:
                test_label = target_ec
                test_pre = pre_ec
                result_pd = pd.DataFrame(columns=["pre_ec", "target_ec"])

                for result_id in range(len(pre_ec)):
                    new = pd.Series({
                        "pre_ec": level4_class[pre_ec[result_id]],
                        "target_ec": level4_class[target_ec[result_id]]
                    })
                    result_pd = result_pd.append(new, ignore_index=True)

                result_pd.to_csv(opt.savepath + str(current_level) + "_fold" + str(flod) + ".csv", index=False)

        acc = accuracy_score(test_label, test_pre)
        F1 = f1_score(test_label, test_pre, average="macro")

        acc_list.append(acc * 100)
        f1_list.append(F1 * 100)

        print('======== Training END ========')
        F_txt.close()
        # ============================================ Training End ========================================
    acc_mean, acc_std = get_mean_std(acc_list)
    f1_mean, f1_std = get_mean_std(f1_list)

    print("Predict Level%d, The Result is mean_acc = %.2f + %.2f mean_f1 = %.2f + %.2f" % (
        current_level, acc_mean, acc_std, f1_mean, f1_std))
    print("Predict Level%d, The Result is mean_acc = %.2f + %.2f mean_f1 = %.2f + %.2f" % (
        current_level, acc_mean, acc_std, f1_mean, f1_std), file=final_T)
    final_T.close()
