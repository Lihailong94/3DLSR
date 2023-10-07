import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models_3d_residual import LSR3D
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr, convert_rgb_to_y, draw_curve

if __name__ == '__main__':

    for _ in range(3):
        print("\033[1;31;40mWarning!!!!       Project name should be set!!!!!!\033[0m")

    parser = argparse.ArgumentParser()
    parser.add_argument('--project-name', type=str, default='012')
    parser.add_argument('--train-file', type=str, default='dataset/3d/s16/DIV2K_800_train_x3_partial.h5')
    parser.add_argument('--eval-file', type=str, default='dataset/test/set5_test_x3.h5')
    parser.add_argument('--outputs-dir', type=str, default='output/3d')
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=300)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    curve_loss = list()
    curve_psnr = list()
    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = LSR3D(scale_factor=args.scale).to(device)
    criterion = nn.L1Loss(reduction='mean')

    # TODO MODEL_3D
    optimizer = optim.Adam([
        {'params': model.first_part.parameters()},
        {'params': model.mid_part.parameters()},
        {'params': model.last_part.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    # TODO MODEL_3D_DRRN
    # optimizer = optim.Adam([
    #     {'params': model.input.parameters()},
    #     {'params': model.conv1.parameters()},
    #     {'params': model.conv2.parameters()},
    #     {'params': model.output.parameters(), 'lr': args.lr * 0.1}
    # ], lr=args.lr)

    # TODO
    # scheduler = lrs.ExponentialLR(optimizer,0.95)
    scheduler = lrs.StepLR(optimizer,step_size=5,gamma=0.9)
    # TODO

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                # TODO
                inputs = inputs.permute(0,1,4,2,3)
                labels = labels.permute(0, 1, 4, 2, 3)
                # TODO

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)
                tem_loss = loss

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

            curve_loss.append(tem_loss.detach().cpu().numpy())
            # TODO
            scheduler.step()
            # TODO

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)       #1 1 7 7 3
            labels = labels.to(device)

            # TODO
            inputs = inputs.permute(0, 1, 4, 2, 3)
            labels = labels.permute(0, 1, 4, 2, 3)
            # TODO

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            # TODO
            preds = preds.permute(0,1,3,4,2)
            preds = convert_rgb_to_y(preds)
            labels = labels.permute(0, 1, 3, 4, 2)
            labels = convert_rgb_to_y(labels)
            # TODO

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
        tem_psnr_ave=epoch_psnr.avg
        curve_psnr.append(tem_psnr_ave.detach().cpu().numpy())
        draw_curve(args.num_epochs-1,curve_psnr,curve_loss,args.project_name)

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
