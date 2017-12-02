import math
import os
import os.path as osp
import torch
import fcn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm
from PIL import Image

def cross_entropy(input, target, weight=None, size_average=True):
# input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


class Tester(object):
    def __init__(self, cuda, model, test_loader,  max_iter, size_average = False):
        self.model = model
        self.cuda = cuda
        self.test_loader = test_loader
        self.size_average = size_average
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0
    
    def test_epoch(self):
        self.model.eval()
        n_class = 21
        img_ind = 1

        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.test_loader), total=len(self.test_loader),
                desc='Test epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.test_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration


            if self.cuda:
                data, target = data, target

            """
            back_propagation
            """
            data, target = Variable(data).cuda(), Variable(target).cuda()
            score = self.model(data)
            n,c,h,w = score.data.shape
            image = score.data.max(1)[1]
            image = image.cpu().numpy().astype(np.uint8)
            image = image.transpose(1,2,0).reshape(h,w)
            image = Image.fromarray(image)
            img_name = ''.join(["saved_img",str(img_ind),'.png'])
            img_ind = img_ind + 1
            image.save(img_name)

            loss = cross_entropy(score, target, size_average = self.size_average)
            print "loss", loss.data[0]
            #loss = loss / len(data)
            if np.isnan(np.float(loss.data[0])):
                raise ValueError('loss is nan while training')
            
            """
            measure accuracy by fcn.utils.label_accuracy_score
            """
            metrics = []
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            for lt, lp in zip(lbl_true, lbl_pred):
                acc, acc_cls, mean_iu, fwavacc = \
                    fcn.utils.label_accuracy_score(
                        [lt], [lp], n_class=n_class)
                metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)

            if self.iteration >= self.max_iter:
                break
    
    def test(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.test_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Test', ncols=80):
            self.epoch = epoch
            self.test_epoch()
            if self.iteration >= self.max_iter:
                break


