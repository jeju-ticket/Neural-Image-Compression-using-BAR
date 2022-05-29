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
import os



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
        "--block",
        type = int,
        default = 256,
        help = "Size of 2Nx2N block - based on 2N"
    )
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


def comrpess_and_decompress(model_mode1, model_mode2, test_dataloader, device, blockSize, lmbda):
    psnr = AverageMeter()
    bpp = AverageMeter()

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

    if(blockSize == 256):
        image_hat_path = "./VER3/256_mode_ver/"
    
    elif(blockSize == 128):
        image_hat_path = "./VER3/128_mode_ver/"
        
    else:
        sys.exit("Invalid block size")   
        

    with torch.no_grad():
        picture_num = 1
        for i_, x in enumerate(test_dataloader):
            print("************************ #", picture_num, " PICTURE BLOCKS ************************")
            isTransposed = False
            x = x.to(device)
            #print(x.size())

            # Image -> Blocks =================================================================
            if(x.size()[2] > x.size()[3]):
                x = torch.transpose(x, 2, 3)
                isTransposed = True

            blocks = []
            block_size = blockSize
            x_stat = 0
            x_des = blockSize
            
            while (x_des <= x.size()[2]):
                y_stat = 0
                y_des = blockSize

                while (y_des <= x.size()[3]):
                    blocks.append(x[:, :, x_stat:x_des, y_stat:y_des]) # blocks.size == 6 when 2N == 256
                    y_stat = y_des
                    y_des += blockSize

                x_stat = x_des
                x_des += blockSize


            # Mode Decision Part ===============================================================
            block_hats = [] # 모드별로 선택된 2nx2n 블록 담는 list
            mode1_bpp_ = 0
            mode1_mse_ = 0
            mode1_psnr_ = 0
            block_number = 1  # 몇번째 블록인지 표시하기 위해서
            for target in blocks:
                target_block = target
                """
                    @ Mode 1
                    @ mini blocks -> NNIC -> a block
                """
                # Mode 1 ---------------------------------------------------------------------------
                ## 1.1. 미니 블록으로 쪼개기            
                mini_blocks = []
                mini_blockSize = blockSize // 2
                
                m_stat = 0
                m_des = mini_blockSize
                
                while (m_des <= blockSize):
                    
                    n_stat = 0
                    n_des = mini_blockSize
                    
                    while(n_des <= blockSize): 
                        mini_blocks.append(target_block[:, :, m_stat:m_des, n_stat:n_des])
                        n_stat = n_des
                        n_des += mini_blockSize

                    m_stat = m_des
                    m_des += mini_blockSize


                ## 1.2. mini blocks를 NNIC에 넣기
                mini_blocks_hat = []
                b_bpp_y = 0
                b_bpp_z = 0
                for b in mini_blocks:

                    # compress
                    compressed = model_mode1.compress(b)  # {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
                    strings = compressed['strings']
                    shape = compressed['shape']

                    # decompress
                    decompressed = model_mode1.decompress(strings, shape)
                    b_hat = decompressed['x_hat'].clamp_(0, 1)
                    mini_blocks_hat.append(b_hat) # 복원된 NxN 블록을 mini_blocks_hat 리스트에 넣음 (나중에 2Nx2N으로 복원하기 위해)


                    ## 1.3. 하나의 N x N 블록에 대해 bitrate 계산
                    b_bpp_y += (len(strings[0][0])) * 8 # bitrate
                    b_bpp_z += (len(strings[1][0])) * 8 # bitrate


                ## 1.4. 복원된 mini blocks를 2N x 2N으로 복원하기
                row1 = torch.cat([mini_blocks_hat[0], mini_blocks_hat[1]], dim=3)
                row2 = torch.cat([mini_blocks_hat[2], mini_blocks_hat[3]], dim=3)
                block_hat = torch.cat([row1, row2], dim=2)

                bpp_y = (b_bpp_y + 1) / (target_block.shape[2] * target_block.shape[3]) # bpp 계산
                bpp_z = (b_bpp_z + 1) / (target_block.shape[2] * target_block.shape[3]) # bpp 계산
                mode1_bpp_ = bpp_y + bpp_z


                mode1_mse_ = (block_hat - target_block).pow(2).mean()
                mode1_psnr_ = 10 * (torch.log(1 * 1 / mode1_mse_) / math.log(10))

                """
                    @ Mode 2
                    @ Downsample -> NNIC -> Upsample
                """
                # # Mode 2 -------------------------------------------------------
                downsampled_block = torch.nn.functional.interpolate(target_block, size=[blockSize // 2, blockSize // 2], mode='bicubic').clamp_(0, 1)
                
                #compress
                compressed = model_mode2.compress(downsampled_block)  # {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
                strings = compressed['strings']
                shape = compressed['shape']

                # decompress
                decompressed = model_mode2.decompress(strings, shape)
                b_hat = decompressed['x_hat']

                upsampled_block = torch.nn.functional.interpolate(b_hat, size=[blockSize, blockSize], mode='bicubic').clamp_(0, 1)

                bpp_y = ((len(strings[0][0])) * 8 + 1) / (target_block.shape[2] * target_block.shape[3])
                # print(bpp_y)
                bpp_z = ((len(strings[1][0])) * 8 + 1) / (target_block.shape[2] * target_block.shape[3])
                # print(bpp_z)
                mode2_bpp_ = bpp_y + bpp_z

                mode2_mse_ = (upsampled_block - target_block).pow(2).mean()
                mode2_psnr_ = 10 * (torch.log(1 * 1 / mode2_mse_) / math.log(10))
                #print("@@ MODE 2 @@")
                #print("mse_ : ", mode2_mse_, "bpp_ : ", mode2_bpp_, "PSNR_ : ", mode2_psnr_)

                """
                    @ Mode Comparison
                """
                # Mode 비교 -------------------------------------------------------------------
                mode1_cost = lmbda * 255 ** 2 * mode1_mse_ + mode1_bpp_
                mode2_cost = lmbda * 255 ** 2 * mode2_mse_ + mode2_bpp_
                
                print("#", block_number, " Block")
                if(mode1_cost < mode2_cost):
                    print("mode1 is better")
                    bpp.update(mode1_bpp_)
                    psnr.update(mode1_psnr_)
                    block_hats.append(block_hat)
                
                else:
                    print("mode2 is better")
                    bpp.update(mode2_bpp_)
                    psnr.update(mode2_psnr_)
                    block_hats.append(upsampled_block)

                block_number += 1
                #print("================================")

            if(blockSize == 256):
                row1 = torch.cat([block_hats[0], block_hats[1], block_hats[2]], dim=3)
                row2 = torch.cat([block_hats[3], block_hats[4], block_hats[5]], dim=3)
                x_hat = torch.cat([row1, row2], dim=2)
                photo = torch.squeeze(x_hat)
                photo = transforms.functional.to_pil_image(photo)
                photo.save(image_hat_path + str(lmbda_to_quality[lmbda]) + "/" +str(picture_num) + ".jpg")
                picture_num += 1   
            
            elif(blockSize == 128):
                row1 = torch.cat([block_hats[0], block_hats[1], block_hats[2], block_hats[3], block_hats[4], block_hats[5]], dim=3)
                row2 = torch.cat([block_hats[6], block_hats[7], block_hats[8], block_hats[9], block_hats[10], block_hats[11]], dim=3)
                row3 = torch.cat([block_hats[12], block_hats[13], block_hats[14], block_hats[15], block_hats[16], block_hats[17]], dim=3)
                row4 = torch.cat([block_hats[18], block_hats[19], block_hats[20], block_hats[21], block_hats[22], block_hats[23]], dim=3)
                x_hat = torch.cat([row1, row2, row3, row4], dim=2)
                photo = torch.squeeze(x_hat)
                photo = transforms.functional.to_pil_image(photo)
                photo.save(image_hat_path + str(lmbda_to_quality[lmbda]) + "/" +str(picture_num) + ".jpg")
                picture_num += 1 

            elif(blockSize == 64):
                row1 = torch.cat([block_hats[0], block_hats[1], block_hats[2], block_hats[3], block_hats[4], block_hats[5],
                                    block_hats[6], block_hats[7], block_hats[8], block_hats[9], block_hats[10], block_hats[11]], dim=3)

                row2 = torch.cat([block_hats[12], block_hats[13], block_hats[14], block_hats[15], block_hats[16], block_hats[17],
                                    block_hats[18], block_hats[19], block_hats[20], block_hats[21], block_hats[22], block_hats[23]], dim=3)

                row3 = torch.cat([block_hats[24], block_hats[25], block_hats[26], block_hats[27], block_hats[28], block_hats[29],
                                    block_hats[30], block_hats[31], block_hats[32], block_hats[33], block_hats[34], block_hats[35]], dim=3)

                row4 = torch.cat([block_hats[36], block_hats[37], block_hats[38], block_hats[39], block_hats[40], block_hats[41],
                                    block_hats[42], block_hats[43], block_hats[44], block_hats[45], block_hats[46], block_hats[47]], dim=3)

                row5 = torch.cat([block_hats[48], block_hats[49], block_hats[50], block_hats[51], block_hats[52], block_hats[53],
                                    block_hats[54], block_hats[55], block_hats[56], block_hats[57], block_hats[58], block_hats[59]], dim=3)

                row6 = torch.cat([block_hats[60], block_hats[61], block_hats[62], block_hats[63], block_hats[64], block_hats[65],
                                    block_hats[66], block_hats[67], block_hats[68], block_hats[69], block_hats[70], block_hats[71]], dim=3)

                row7 = torch.cat([block_hats[72], block_hats[73], block_hats[74], block_hats[75], block_hats[76], block_hats[77],
                                    block_hats[78], block_hats[79], block_hats[80], block_hats[81], block_hats[82], block_hats[83]], dim=3)

                row8 = torch.cat([block_hats[84], block_hats[85], block_hats[86], block_hats[87], block_hats[88], block_hats[89],
                                    block_hats[90], block_hats[91], block_hats[92], block_hats[93], block_hats[94], block_hats[95]], dim=3)

                x_hat = torch.cat([row1, row2, row3, row4, row5, row6, row7, row8], dim = 2)
                photo = torch.squeeze(x_hat)
                photo = transforms.functional.to_pil_image(photo)
                photo.save(image_hat_path + str(lmbda_to_quality[lmbda]) + "/" +str(picture_num) + ".jpg")
                picture_num += 1 


            
            
            else:
                sys.exit("Invalid Block Size")   
   


    print(
    f"\tTest PSNR: {psnr.avg:.3f} |"
    f"\tTest BPP: {bpp.avg:.3f} |"
    )

    if(blockSize == 256) :
        with open("./VER3/result/256_RD.txt", 'a') as f:
            f.write(
            f"\tLmbda: {lmbda_to_quality[lmbda]} |"
            f"\tPSNR: {psnr.avg:.3f} |"
            f"\tBPP: {bpp.avg:.3f} | \n")

    elif(blockSize == 128): 
        with open("./VER3/result/128_RD.txt", 'a') as f:
            f.write(
            f"\tLmbda: {lmbda_to_quality[lmbda]} |"
            f"\tPSNR: {psnr.avg:.3f} |"
            f"\tBPP: {bpp.avg:.3f} | \n")

    else:
        sys.exit("Invalid Block Size")   

    


def main(argv):
    torch.backends.cudnn.deterministic = True
    
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    qualities_to_lambda = {
        1: 0.0018,
        2: 0.0035,
        3: 0.0067,
        4: 0.0130,
        5: 0.0250,
        6: 0.0483,
        7: 0.0932,
        8: 0.1800
    }
    
    device = "cuda"
    for q in range(1,7):
        model_mode1 = load_model(args.model, metric="mse", quality=q, pretrained=True).to(device).eval()
        model_mode2 = load_model(args.model, metric="mse", quality=q+2, pretrained=True).to(device).eval()
        lmbda = qualities_to_lambda[q]

        test_dataloader = build_dataset(args)
        print(f"Quality : {q} ")
        comrpess_and_decompress(model_mode1, model_mode2, test_dataloader, device, args.block, lmbda)


if __name__ == "__main__":
    main(sys.argv[1:])