import math

import torch
from torch import nn


class LSR3D(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(LSR3D, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv3d(num_channels, d, kernel_size=(3, 5, 5), padding=(1, 5 // 2, 5 // 2)),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv3d(d, s, kernel_size=1), nn.PReLU(s)]

        self.mid_part.extend([nn.Conv3d(s, s, kernel_size=3, padding=3 // 2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv3d(s, s, kernel_size=3, padding=3 // 2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv3d(s, s, kernel_size=3, padding=3 // 2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv3d(s, s, kernel_size=3, padding=3 // 2), nn.PReLU(s)])

        # TODO deConv
        self.mid_part.extend([nn.Conv3d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        # TODO ACTIVATE IT WHEN USE 3 SCALE
        self.last_part = nn.ConvTranspose3d(d, num_channels, kernel_size=(3, 9, 9), stride=(1,scale_factor,scale_factor),
                                            padding=(1, 9 // 2, 9 //2),
                                            output_padding=(0,2,2))
        # TODO ACTIVATE IT WHEN USE 4 SCALE
        # self.last_part = nn.ConvTranspose3d(d, num_channels, kernel_size=(3, 9, 9), stride=(1, 4, 4),
        #                                     padding=(1, 9 // 2, 9 // 2),
        #                                     output_padding=(0,3,3))
        # # TODO ACTIVATE IT WHEN USE 2 SCALE
        # self.last_part = nn.ConvTranspose3d(d, num_channels, kernel_size=(3, 9, 9), stride=(1, 2, 2),
        #                                     padding=(1, 9 // 2, 9 // 2),
        #                                     output_padding=(0,1,1))
        # TODO PixelShuffle
        # self.mid_part.extend([nn.Conv3d(s, scale_factor*scale_factor, kernel_size=1), nn.PReLU(scale_factor*scale_factor)])
        # self.mid_part = nn.Sequential(*self.mid_part)
        # self.last_part = nn.PixelShuffle(3)
        # TODO END

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)





