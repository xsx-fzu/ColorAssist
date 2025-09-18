import time
import torch
import numpy as np
import torch.nn.functional as F
import warnings

def gaussian_filter(input, win, stride=1):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            if stride == 1:
                out = conv(out, weight=win.transpose(2 + i, -1), stride=stride, padding=0, groups=C)
            else:
                S = [(stride, 1), (1, stride)]
                out = conv(out, weight=win.transpose(2 + i, -1), stride=S[i], padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )
    return out

def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)

def colorInfoLoss(X, Y, size_average=True, win_size=11, win_sigma=5, win=None):

    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    if not X.shape[1] == Y.shape[1]:
        raise ValueError(f"Input images should be color images.")
    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.type() == Y.type():
        raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    loss = _colorLoss(X, Y, win=win)

    return loss

def _colorLoss(X, Y, win):
    r""" Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images in CIE La*b* space
        Y (torch.Tensor): images in CIE La*b* space
        win (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: color information loss results.
    """
    assert X.shape[1] == Y.shape[1] == 3
    # X = X[:, 1:, :, :]
    # Y = Y[:, 1:, :, :]
    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win, stride=win.shape[3]).reshape(-1)  # 即mean
    mu2 = gaussian_filter(Y, win, stride=win.shape[3]).reshape(-1)

    res = torch.abs(mu1 - mu2)
    return res.mean()


def global_contrast_img_l1(img,img2,points_number=5):
    img = img.permute(0, 2, 3, 1)
    img2 = img2.permute(0,2,3,1)
    hight, width = img.shape[1],img.shape[2]
    select_points = torch.tensor(np.zeros((img.size(0), *(1,points_number,1))))
    rand_hight = torch.randint(0, hight, (points_number,))
    rand_width = torch.randint(0, width, (points_number,))
    rand_hight1 = torch.randint(0, hight, (points_number,))
    rand_width1= torch.randint(0, width, (points_number,))
    img_points1 = img[:,rand_width,rand_hight,:]
    img_points2 = img[:, rand_width1, rand_hight1, :]
    img1_diff = (img_points1 - img_points2) #*(img_points1 - img_points2)
    img1_diff = torch.sum(torch.abs(img1_diff), 2)
    img2_points1 = img2[:,rand_width,rand_hight,:]
    img2_points2 = img2[:, rand_width1, rand_hight1, :]
    img2_diff = (img2_points1 - img2_points2) #* (img2_points1 - img2_points2)
    img2_diff = torch.sum(torch.abs(img2_diff), 2)
    return img1_diff,img2_diff

def calculate_contrast_oneimg_l1(img,window_size):
    img = img.permute(0, 2, 3, 1)
    x_diff = img[:, window_size:128 - window_size, window_size:128 - window_size]
    x = x_diff
    #print('2222',x.shape)
    start = time.time()
    flag = 0
    for i in range(-window_size, window_size + 1):
        for j in range(-window_size, window_size + 1):
            if (i == -window_size) and (j == -window_size):
                img_diff = x_diff - img[:, window_size + i:128 - window_size + i, window_size + j:128 - window_size + j]
                img_diff = torch.sum(torch.abs(img_diff), 3)
                #print(img_diff,img_diff.size())
                img_diff = torch.unsqueeze(img_diff, 3)
                x = img_diff
                flag += 1
            elif (i == 0) and (j == 0):
                continue
            else:
                # print(img_diff.shape)
                flag += 1
                # print(i, j, flag)
                img_diff = x_diff - img[:, window_size + i:128 - window_size + i, window_size + j:128 - window_size + j]
                img_diff = torch.sum(torch.abs(img_diff), 3)
                img_diff = torch.unsqueeze(img_diff, 3)
                x = torch.cat((x, img_diff), 3)
    nrand = np.array([i for i in range(120)])
    trand = torch.from_numpy(nrand).type(torch.long)
    return abs(x[:, :, :, trand])
