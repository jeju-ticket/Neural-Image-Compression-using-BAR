
# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from collections import defaultdict
import argparse
import math
import random
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.zoo import image_models
from dataset import Kodak24Dataset
from load_model import load_model



def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    '''
    "bmshj2018-factorized": bmshj2018_factorized,
    "bmshj2018-hyperprior": bmshj2018_hyperprior,
    "mbt2018-mean": mbt2018_mean,
    "mbt2018": mbt2018,
    "cheng2020-anchor": cheng2020_anchor,
    "cheng2020-attn": cheng2020_attn,
    '''
    parser.add_argument(
        "-m",
        "--model",
        default="mbt2018-mean",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    # parser.add_argument(
    #     "-q", "--quality", type=int, default=0, help="quality of the model"
    # )

    parser.add_argument(
        "-d", "--dataset", type=str, default='../data', help="Training dataset"
    )
    parser.add_argument(
        "-save_dir", "--save_dir", type=str, default='save/', help="save_dir"
    )
    parser.add_argument(
        "-log_dir", "--log_dir", type=str, default='log/', help="log_dir"
    )
    parser.add_argument(
        "-total_step",
        "--total_step",
        default=5000000,    # infinite (we can use early stop)
        type=int,
        help="total_step (default: %(default)s)",
    )
    parser.add_argument(
        "-test_step",
        "--test_step",
        default=5000,
        type=int,
        help="test_step (default: %(default)s)",
    )
    parser.add_argument(
        "-save_step",
        "--save_step",
        default=100000,
        type=int,
        help="save_step (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    # parser.add_argument(
    #     "--lambda",
    #     dest="lmbda",
    #     type=float,
    #     default=1e-2,
    #     # quality   :     1        2        3        4        5        6       7       8
    #     # lambda    :  0.0018   0.0035   0.0067   0.0130   0.0250   0.0483  0.0932   0.1800
    #     help="Bit-rate distortion parameter (default: %(default)s)",
    # )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", default=True, help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=123, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )

    parser.add_argument(
        "--block",
        type = int,
        default = 256,
        help = "Size of 2Nx2N block - based on 2N"
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def build_dataset(args):
    # Warning, the order of the transform composition should be kept.
    kodak_transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    kodak_dataset = Kodak24Dataset(
        args.dataset,
        transform=kodak_transform,
    )

    kodak_dataloader = DataLoader(
        kodak_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    return kodak_dataloader


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def comrpess_and_decompress(model, test_dataloader, device, blockSize, quality):

    lmbda_to_quality = {
        0.0018 :1,
        0.0035 :2,
        0.0067 :3,
        0.0130 :4,
        0.0250 :5,
        0.0483 :6,
        0.0932 :7,
        0.1800 :8
    }

    if(blockSize == 128):
        image_hat_path = "./block-based/128/"
    
    else:
        image_hat_path = "./block-based/256/"

    psnr = AverageMeter()
    bpp = AverageMeter()



    with torch.no_grad():
        picture_num = 1 
        for i_, x in enumerate(test_dataloader):
            
            isTransposed = False
            x = x.to(device)
            #print(x.size())

            
            if(x.size()[2] > x.size()[3]):
                x = torch.transpose(x, 2, 3)
                #print("Transposed : ", x.size())
                isTransposed = True

            blocks = []
            block_size = blockSize
            x_stat = 0
            x_des = block_size
            

            while (x_des <= x.size()[2]):
                y_stat = 0
                y_des = block_size

                while (y_des <= x.size()[3]):
                    blocks.append(x[:, :, x_stat:x_des, y_stat:y_des])
                    y_stat = y_des
                    y_des += block_size

                x_stat = x_des
                x_des += block_size

            
            block_hats = [] # ????????? 2nx2n ?????? ?????? list
            mode1_bpp_ = 0
            mode1_mse_ = 0
            mode1_psnr_ = 0
            block_number = 1  # ????????? ???????????? ???????????? ?????????
            for target in blocks:
                target_block = target
                """
                    @ Mode 1
                    @ mini blocks -> NNIC -> a block
                """
                # Mode 1 ---------------------------------------------------------------------------
                ## 1.1. ?????? ???????????? ?????????            
                mini_blocks = []
                mini_block_size = block_size // 2
                
                m_stat = 0
                m_des = mini_block_size
                
                while (m_des <= block_size):
                    
                    n_stat = 0
                    n_des = mini_block_size
                    
                    while(n_des <= block_size): 
                        mini_blocks.append(target_block[:, :, m_stat:m_des, n_stat:n_des])
                        n_stat = n_des
                        n_des += mini_block_size

                    m_stat = m_des
                    m_des += mini_block_size


                ## 1.2. mini blocks??? NNIC??? ??????
                mini_blocks_hat = []
                b_bpp_y = 0
                b_bpp_z = 0
                for b in mini_blocks:

                    # compress
                    compressed = model.compress(b)  # {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
                    strings = compressed['strings']
                    shape = compressed['shape']

                    # decompress
                    decompressed = model.decompress(strings, shape)
                    b_hat = decompressed['x_hat'].clamp_(0, 1)
                    mini_blocks_hat.append(b_hat) # ????????? NxN ????????? mini_blocks_hat ???????????? ?????? (????????? 2Nx2N?????? ???????????? ??????)


                    ## 1.3. ????????? N x N ????????? ?????? bitrate ??????
                    b_bpp_y += (len(strings[0][0])) * 8 # bitrate
                    b_bpp_z += (len(strings[1][0])) * 8 # bitrate


                ## 1.4. ????????? mini blocks??? 2N x 2N?????? ????????????
                row1 = torch.cat([mini_blocks_hat[0], mini_blocks_hat[1]], dim=3)
                row2 = torch.cat([mini_blocks_hat[2], mini_blocks_hat[3]], dim=3)
                block_hat = torch.cat([row1, row2], dim=2)
                block_hats.append(block_hat)

                bpp_y = b_bpp_y / (target_block.shape[2] * target_block.shape[3]) # bpp ??????
                bpp_z = b_bpp_z / (target_block.shape[2] * target_block.shape[3]) # bpp ??????
                mode1_bpp_ = bpp_y + bpp_z


                mode1_mse_ = (block_hat - target_block).pow(2).mean()
                mode1_psnr_ = 10 * (torch.log(1 * 1 / mode1_mse_) / math.log(10))

                bpp.update(mode1_bpp_)
                psnr.update(mode1_psnr_)

            if(blockSize == 256):
                row1 = torch.cat([block_hats[0], block_hats[1], block_hats[2]], dim=3)
                row2 = torch.cat([block_hats[3], block_hats[4], block_hats[5]], dim=3)
                x_hat = torch.cat([row1, row2], dim=2)
                photo = torch.squeeze(x_hat)
                photo = transforms.functional.to_pil_image(photo)
                photo.save(image_hat_path + str(quality) + "/" +str(picture_num) + ".jpg")
                picture_num += 1   
            
            else:
                row1 = torch.cat([block_hats[0], block_hats[1], block_hats[2], block_hats[3], block_hats[4], block_hats[5]], dim=3)
                row2 = torch.cat([block_hats[6], block_hats[7], block_hats[8], block_hats[9], block_hats[10], block_hats[11]], dim=3)
                row3 = torch.cat([block_hats[12], block_hats[13], block_hats[14], block_hats[15], block_hats[16], block_hats[17]], dim=3)
                row4 = torch.cat([block_hats[18], block_hats[19], block_hats[20], block_hats[21], block_hats[22], block_hats[23]], dim=3)
                x_hat = torch.cat([row1, row2, row3, row4], dim=2)
                photo = torch.squeeze(x_hat)
                photo = transforms.functional.to_pil_image(photo)
                photo.save(image_hat_path + str(quality) + "/" +str(picture_num) + ".jpg")
                picture_num += 1 
        

    print(
    f"\tTest PSNR: {psnr.avg:.3f} |"
    f"\tTest BPP: {bpp.avg:.3f} |"
    )

    if(block_size == 128):
        with open("./block-based/128_result.txt", 'a') as f:
            f.write(
            f"\tQuality: {quality} |"
            f"\tPSNR: {psnr.avg:.3f} |"
            f"\tBPP: {bpp.avg:.3f} | \n")
    
    elif(block_size == 256):
        with open("./block-based/256_result.txt", 'a') as f:
            f.write(
            f"\tQuality: {quality} |"
            f"\tPSNR: {psnr.avg:.3f} |"
            f"\tBPP: {bpp.avg:.3f} | \n")
    else:
        sys.exit("Invalid Block")



def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = "cuda"
    for q in range(1,9):
        model = load_model(args.model, metric="mse", quality=q, pretrained=True).to(device).eval()

        if args.checkpoint:  # load from previous checkpoint
            print("Loading", args.checkpoint)
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])

        test_dataloader = build_dataset(args)
        print(f"Quality : {q} ")
        comrpess_and_decompress(model, test_dataloader, device, args.block, q)


if __name__ == "__main__":
    main(sys.argv[1:])
