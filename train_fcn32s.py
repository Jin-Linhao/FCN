import os
import os.path as osp
import torchvision.models as models
import torch
import torch.nn as nn
import fcn
from fcn.datasets.mli import ImageList
from fcn import Trainer
import torchvision.transforms as transforms

configurations = {
    # same configuration as original fcn32s
    1: dict(
	max_iteration = 10000,
	lr = 1.0e-10,
	momentum = 0.99,
	weight_decay = 0.0005,
	interval_validate = 4000,
    )
}

def get_parameters(model, bias):
    std_module = (	
        nn.ReLU,
        nn.Sequential,
        nn.MaxPool2d,
        nn.Dropout2d,
        fcn.models.FCN32s,
        # fcn.models.FCN16s,
        # fcn.models.FCN8s
    )
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, std_module):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))

def main():
    print 'start'
    file = '/home/yaohuaxu1/fcn32s_from_caffe.pth'
    model = fcn.models.FCN32s()
    model.load_state_dict(torch.load(file))
    model = model.cuda()
    print "start loading"
    model.score_fr = nn.Conv2d(4096, 2, 1)
    model.upscore = nn.ConvTranspose2d(2,2,64, stride=32, bias=False)
    for m in model.modules():
        print m
    model.upscore = nn.ConvTranspose2d(2,2,64, stride=32,bias=False)
    train_dataloader = torch.utils.data.DataLoader(
        ImageList(fileList="/home/yaohuaxu1/FCN/train.txt",
                  transform=transforms.Compose([
                      transforms.ToTensor(), ])),
        shuffle=False,
        num_workers=8,
        batch_size=1)
    start_epoch = 0
    start_iteration = 0
    cfg = configurations
    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True),
             'lr': cfg[1]['lr'] * 2, 'weight_decay': 0},
        ],
        lr=cfg[1]['lr'],
        momentum=cfg[1]['momentum'],
        weight_decay=cfg[1]['weight_decay'])

    print "going into trainer"
    trainer = Trainer(cuda=False,
                      model=model,
                      optimizer=optim,
                      train_loader=train_dataloader,
                      val_loader=train_dataloader,
                      max_iter=cfg[1]['max_iteration'],
                      size_average=False
                        )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()
    torch.save(model.state_dict(), '/home/yaohuaxu1/FCN')


if __name__ == '__main__':
    main()
