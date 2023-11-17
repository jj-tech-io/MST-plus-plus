import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
import datetime
from hsi_dataset import TrainDataset, ValidDataset
from architecture import model_generator
from utils import AverageMeter, initialize_logger, save_checkpoint, Loss_MRAE, Loss_RMSE, Loss_PSNR
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--batch_size", type=int, default=8, help="batch size")
parser.add_argument("--end_epoch", type=int, default=100, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--outf", type=str, default='./exp/mst_plus_plus/', help='path to log files')
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

# Load dataset
train_data = TrainDataset(opt.train_spec, opt.train_rgb, opt.patch_size, True, False, opt.stride)
val_data = ValidDataset(opt.test_spec, opt.test_rgb, False)
print(f"\nLoading dataset... \nIteration per epoch: {len(train_data)} \nValidation set samples: {len(val_data)}")

# Loss functions
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()

# Model
model = model_generator(opt.method, opt.pretrained_model_path)
if torch.cuda.is_available():
    model = model.cuda()
    criterion_mrae = criterion_mrae.cuda()
    criterion_rmse = criterion_rmse.cuda()
    criterion_psnr = criterion_psnr.cuda()

# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_data)*opt.end_epoch, eta_min=1e-6)

# Logging
log_dir = os.path.join(opt.outf, 'train.log')
logger = initialize_logger(log_dir)

# Resume from checkpoint
if opt.pretrained_model_path and os.path.isfile(opt.pretrained_model_path):
    print(f"=> Loading checkpoint '{opt.pretrained_model_path}'")
    checkpoint = torch.load(opt.pretrained_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Training function
def main():
    cudnn.benchmark = True
    record_mrae_loss = 10
    for epoch in range(opt.end_epoch):
        model.train()
        losses = AverageMeter()
        train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        for i, (images, labels) in enumerate(train_loader):
            # Training step
            labels = labels.cuda()
            images = images.cuda()
            optimizer.zero_grad()
            output = model(images)
            loss = criterion_mrae(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.update(loss.item())

            # Validation and logging
            if i % 5 == 0:
                mrae_loss, rmse_loss, psnr_loss = validate(val_loader, model)
                print(f"Epoch [{epoch+1}/{opt.end_epoch}], Iter [{i+1}/{len(train_loader)}], MRAE: {mrae_loss}, RMSE: {rmse_loss}, PSNR: {psnr_loss}")
                logger.info(f"Epoch [{epoch+1}/{opt.end_epoch}], Iter [{i+1}/{len(train_loader)}], MRAE: {mrae_loss}, RMSE: {rmse_loss}, PSNR: {psnr_loss}")

                # Save checkpoint
                if mrae_loss < record_mrae_loss:
                    record_mrae_loss = mrae_loss
                    save_checkpoint(opt.outf, epoch, i, model, optimizer)

    return model

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
    trained_model = main()
    save_checkpoint(opt.outf, opt.end_epoch, len(train_data)*opt.end_epoch, trained_model, optimizer)
    print(f"Model training completed. Torch version: {torch.__version__}")