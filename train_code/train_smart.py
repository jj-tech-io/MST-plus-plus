import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
import datetime
from hsi_dataset import TrainDataset, ValidDataset
from architecture import model_generator
import argparse
from torch.autograd import Variable

utils = 'utils'
import sys
sys.path.append(utils)
#save_checkpoint = utils.save_checkpoint

from architecture import *
import utils

from utils import AverageMeter, initialize_logger, record_loss, time2file_name, Loss_MRAE, Loss_RMSE, Loss_PSNR
import datetime
# Argument parsing
parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--batch_size", type=int, default=20, help="batch size")
parser.add_argument("--end_epoch", type=int, default=10, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--outf", type=str, default='.exp/trained', help='path to log files')
parser.add_argument("--train_rgb", type=str, default='./dataset/Train_RGB/', help='path to training RGB images')
parser.add_argument("--train_spec", type=str, default='./dataset/Train_Spec/', help='path to training spectral images')
parser.add_argument("--test_rgb", type=str, default='./dataset/Val_RGB/', help='path to testing RGB images')
parser.add_argument("--test_spec", type=str, default='./dataset/Val_Spec/', help='path to testing spectral images')
parser.add_argument("--data_root", type=str, default='./dataset/', help='path to dataset')
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpu_id", type=str, default='0', help='GPU ID')
opt = parser.parse_args()

# Environment setup
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
# Global Variables
model, train_loader, val_loader, optimizer, scheduler, logger, total_iteration, criterion_mrae, criterion_rmse, criterion_psnr = None, None, None, None, None, None, None, None, None, None
def setup():
    global model, train_loader, val_loader, optimizer, scheduler, logger, total_iteration, criterion_mrae, criterion_rmse, criterion_psnr
    # Load dataset
    train_data = TrainDataset(opt.train_spec, opt.train_rgb, opt.patch_size, True, False, opt.stride)
    val_data = ValidDataset(opt.test_spec, opt.test_rgb, False)

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # iterations
    per_epoch_iteration = 100
    total_iteration = per_epoch_iteration*opt.end_epoch

    # loss function
    criterion_mrae = Loss_MRAE()
    criterion_rmse = Loss_RMSE()
    criterion_psnr = Loss_PSNR()

    # model
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method
    model = model_generator(method, pretrained_model_path).cuda()
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))

    # output path
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    opt.outf = opt.outf + date_time
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)

    if torch.cuda.is_available():
        model.cuda()
        criterion_mrae.cuda()
        criterion_rmse.cuda()
        criterion_psnr.cuda()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)

    # logging
    log_dir = os.path.join(opt.outf, 'train.log')
    logger = initialize_logger(log_dir)

    # Resume
    resume_file = opt.pretrained_model_path
    if resume_file is not None:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
# Training function
def main():
    global model, train_loader, val_loader, optimizer, scheduler, logger
    setup()
    cudnn.benchmark = True
    iteration = 0
    record_mrae_loss = 1000
    while iteration<total_iteration:
        model.train()
        losses = AverageMeter()
        for i, (images, labels) in enumerate(train_loader):
            labels = labels.cuda()
            images = images.cuda()
            images = Variable(images)
            labels = Variable(labels)
            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()
            output = model(images)
            loss = criterion_mrae(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.update(loss.data)
            iteration = iteration+1
            if iteration % 20 == 0:
                print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f'
                      % (iteration, total_iteration, lr, losses.avg))
            if iteration % 1000 == 0:
                mrae_loss, rmse_loss, psnr_loss = validate(val_loader, model)
                print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}')
                # Save model
                if abs(mrae_loss - record_mrae_loss) < 0.01 or mrae_loss < record_mrae_loss or iteration % 5000 == 0:
                    print(f'Saving to {opt.outf}')
                    save_checkpoint(opt.outf, (iteration // 1000), iteration, model, optimizer)
                    if mrae_loss < record_mrae_loss:
                        record_mrae_loss = mrae_loss
                # print loss
                print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
                      "Test RMSE: %.9f, Test PSNR: %.9f " % (iteration, iteration//1000, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss))
                logger.info(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train Loss: %.9f, Test MRAE: %.9f, "
                      "Test RMSE: %.9f, Test PSNR: %.9f " % (iteration, iteration//1000, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss))
    return 0

def validate(val_loader, model):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    for input, target in val_loader:
        input, target = input.cuda(), target.cuda()
        with torch.no_grad():
            output = model(input)
            loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_rmse = criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            losses_mrae.update(loss_mrae.item())
            losses_rmse.update(loss_rmse.item())
            losses_psnr.update(loss_psnr.item())
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg

if __name__ == '__main__':
    main()
    print(torch.__version__)