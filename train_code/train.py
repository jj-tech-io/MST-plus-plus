import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from hsi_dataset import TrainDataset, ValidDataset
from architecture import *
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, \
    time2file_name, Loss_MRAE, Loss_RMSE, Loss_PSNR
import datetime
import hsi_dataset
import importlib
#reloaad hsi_dataset
# importlib.reload(hsi_dataset)
parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--batch_size", type=int, default=8, help="batch size")
parser.add_argument("--end_epoch", type=int, default=10, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--outf", type=str, default='./exp/mst_plus_plus/', help='path log files')
parser.add_argument("--train_rgb", type=str, default='./dataset/Train_RGB/', help='path to training rgb images')
parser.add_argument("--train_spec", type=str, default='./dataset/Train_Spec/', help='path to training spec images')
parser.add_argument("--test_rgb", type=str, default='./dataset/Val_RGB/', help='path to testing rgb images')
parser.add_argument("--test_spec", type=str, default='./dataset/Val_Spec/', help='path to testing spec images')

parser.add_argument("--data_root", type=str, default='./dataset/', help='path to dataset')
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpu_id", type=str, default='0', help='path log files')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# load dataset
print("\nloading dataset ...")
#     def __init__(self, train_spec_path, train_rgb_path, crop_size, arg=True, bgr2rgb=True, stride=8):
train_data = TrainDataset(train_spec_path=opt.train_spec, train_rgb_path=opt.train_rgb, crop_size=opt.patch_size, arg=True, bgr2rgb=False, stride=opt.stride)
print(f"Iteration per epoch: {len(train_data)}")
# val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=False)
val_data = ValidDataset(test_spec_path=opt.test_spec, test_rgb_path=opt.test_rgb, bgr2rgb=False)
print("Validation set samples: ", len(val_data))

# iterations
per_epoch_iteration = 10
total_iteration = per_epoch_iteration*opt.end_epoch

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()

# model
pretrained_model_path = opt.pretrained_model_path
method = opt.method
model = None
#check if cuda is available
if torch.cuda.is_available():
    print('cuda is available')

    model = model_generator(method, pretrained_model_path).cuda()
else:
    model = model_generator(method, pretrained_model_path).cpu()
print('Parameters number is ', sum(param.numel() for param in model.parameters()))

# output path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
opt.outf = opt.outf + date_time
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)
#print torch.__version__ cuda version
print(torch.__version__)
print(torch.version.cuda)
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

def main():
    cudnn.benchmark = True
    record_mrae_loss = 10
    num_epochs = total_iteration // len(train_data) # Assuming total_iteration is total number of batches to process
    
    for epoch in range(num_epochs):
        model.train()
        losses = AverageMeter()
        train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        for i, (images, labels) in enumerate(train_loader):
            iteration = epoch * len(train_loader) + i  # Compute the global step (iteration) number

            # Training step
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

            if iteration % 5 == 0:
                print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f' % (iteration, total_iteration, lr, losses.avg))

            if iteration % 5 == 0:
                mrae_loss, rmse_loss, psnr_loss = validate(val_loader, model)
                print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}')
                
                # Checkpoint saving logic here
                if torch.abs(mrae_loss - record_mrae_loss) < 0.01 or mrae_loss < record_mrae_loss or iteration % 5000 == 0:
                    print(f'Saving to {opt.outf}')
                    save_checkpoint(opt.outf, epoch, iteration, model, optimizer)
                    if mrae_loss < record_mrae_loss:
                        record_mrae_loss = mrae_loss

                # Logging
                print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
                      "Test RMSE: %.9f, Test PSNR: %.9f " % (iteration, epoch, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss))
                logger.info(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train Loss: %.9f, Test MRAE: %.9f, "
                      "Test RMSE: %.9f, Test PSNR: %.9f " % (iteration, epoch, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss))

    return model
# Validate
def validate(val_loader, model):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)
            loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_rmse = criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg

if __name__ == '__main__':
    model = main()
    # Save model
    save_checkpoint(opt.outf, opt.end_epoch, total_iteration, model, optimizer)
    
    
    #print torch.__version__ cuda version
    print(torch.__version__)