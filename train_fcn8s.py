import argparse
import os
import os.path as osp
import torch
from train_fcn32s import get_parameters
from fcn.datasets.mli import ImageList
from fcn.datasets.mli import ImageTest
from fcn import Trainer
import torchvision.transforms as transforms
from fcn.models import FCN16s
from fcn.models import FCN32s

import os
import torchvision
import torch
import torch.nn as nn
import fcn
from fcn.datasets.mli import ImageList
from fcn import Trainer
import argparse
from fcn import Tester

configurations = {
    # same configuration as original fcn32s
    1: dict(
        max_iteration=10000,
        lr=1.0e-5,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=4000,
    ),
    2: dict(
        max_iteration=1,
        lr=1.0e-5,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=4000,
    )

}


def main():
    file = '/home/yaohuaxu1/FCN/fcn16s_model'
    model = fcn.models.FCN8s(n_class=2)
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument("file", type=str)
    args = parser.parse_args()
    if args.save:
        fcn16s = fcn.models.FCN16s(n_class=2)
        fcn16s_state_dict = torch.load(file)
        fcn16s.load_state_dict(fcn16s_state_dict)
        model.copy_params_from_fcn16s(fcn16s)
        model = model.cuda()
        train_dataloader = torch.utils.data.DataLoader(
            ImageList(fileList="/home/yaohuaxu1/FCN/train.txt",
                      transform=transforms.Compose([
                          transforms.ToTensor(), ])),
            shuffle=True,
            num_workers=8,
            batch_size=1)
        start_epoch = 0
        start_iteration = 0
        cfg = configurations
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        trainer = Trainer(cuda=True,
                          model=model,
                          optimizer=optimizer,
                          train_loader=train_dataloader,
                          val_loader=train_dataloader,
                          max_iter=cfg[1]['max_iteration'],
                          size_average=False
                          )
        trainer.epoch = start_epoch
        trainer.iteration = start_iteration
        trainer.train()
        torch.save(model.state_dict(), f=args.file)
    if args.load:
        model.load_state_dict(torch.load(f=args.file))
        model = model.cuda()
        test_dataloader = torch.utils.data.DataLoader(
            ImageTest(fileList="/home/yaohuaxu1/FCN/test.txt",
                      transform=transforms.Compose([
                          transforms.ToTensor(), ])),
            shuffle=False,
            num_workers=8,
            batch_size=1)
        start_epoch = 0
        start_iteration = 0
        cfg = configurations
        tester = Tester(cuda=True,
                        model=model,
                        test_loader= test_dataloader,
                        max_iter=cfg[2]['max_iteration'],
                        size_average=False
                        )
        tester.epoch = start_epoch
        tester.iteration = start_iteration
        tester.test()


if __name__ == '__main__':
    main()
