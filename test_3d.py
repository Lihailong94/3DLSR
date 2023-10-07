import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models_3d_residual_final import LSR3D
from utils import convert_ycbcr_to_rgb, calc_psnr, convert_rgb_to_y


def preprocess(img, device):
    img = np.array(img).astype(np.float32)
    # ycbcr = convert_rgb_to_ycbcr(img)
    x = img
    x /= 255.
    x = torch.from_numpy(x).to(device)
    x = x.unsqueeze(0).unsqueeze(0)
    return x, x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default='output/3d/x3/best (34.14).pth')
    parser.add_argument('--image-file', type=str, default='/home/lhl/Desktop/LHL/Datasets/Set5/baby.png')
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = LSR3D(scale_factor=args.scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
    bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))
    hr.save(args.image_file.replace('.', '_HR_x{}.'.format(args.scale)))

    lr, _ = preprocess(lr, device)
    lr = lr.permute(0,1,4,2,3)
    hr, _ = preprocess(hr, device)
    _, ycbcr = preprocess(bicubic, device)
    hr = convert_rgb_to_y(hr.cpu())

    with torch.no_grad():

        preds = model(lr).clamp(0.0, 1.0)
        preds = preds.permute(0,1,3,4,2)
        preds_psnr = convert_rgb_to_y(preds.cpu())

    psnr = calc_psnr(hr, preds_psnr)
    print('PSNR: {:.2f}'.format(psnr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    # output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    # output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(preds.astype(np.uint8))                     # H*W*3
    output.save(args.image_file.replace('.', '_RCAN_x{}.'.format(args.scale)))



