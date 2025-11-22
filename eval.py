from utils import MSE, PSNR
import torch
from tqdm import tqdm
from utils import count_percentage_super
from torch import optim, nn
import math

import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import os
import torch
from torchvision.utils import save_image
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def EVAL_proposed(model, data_loader, data_loader2 , device, config, epoch):

    model.eval()

    psnr_total_leg = 0
    psnr_total_eve= 0
    mse_total_leg = 0
    mse_total_eve = 0

    total = 0

    for batch_idx, ((data, target_coarse),(data_mnist)) in enumerate(tqdm(zip(data_loader, data_loader2))):
        data, target_coarse , data_mnist = data.to(device), target_coarse.to(device), data_mnist.to(device)

        total += len(target_coarse)
        with torch.no_grad():
            _, z, z_hat_leg, z_hat_eve, rec_leg, rec_eve , _ , _ , _= model(data,data_mnist)

        psnr_bad = PSNR(data, rec_leg)
        psnr_good = PSNR(data, rec_eve)

        mse_bad = MSE(data, rec_leg)
        mse_good = MSE(data, rec_eve)

        psnr_total_leg += psnr_bad
        psnr_total_eve += psnr_good
        mse_total_leg += mse_bad
        mse_total_eve += mse_good

    psnr_total_leg /= total
    psnr_total_eve /= total
    mse_total_leg /= total
    mse_total_eve /= total

    return 0, 0, psnr_total_leg, psnr_total_eve, mse_total_leg, mse_total_eve


