import numpy as np
import torch

def machado_2009_matrices(deficiency, severity):
    if deficiency == 'PROTAN':
        switch_severity = {
            0: np.array(
                [[1.000000, 0.000000, -0.000000], [0.000000, 1.000000, 0.000000], [-0.000000, -0.000000, 1.000000]]),
            1: np.array(
                [[0.856167, 0.182038, -0.038205], [0.029342, 0.955115, 0.015544], [-0.002880, -0.001563, 1.004443]]),
            2: np.array(
                [[0.734766, 0.334872, -0.069637], [0.051840, 0.919198, 0.028963], [-0.004928, -0.004209, 1.009137]]),
            3: np.array(
                [[0.630323, 0.465641, -0.095964], [0.069181, 0.890046, 0.040773], [-0.006308, -0.007724, 1.014032]]),
            4: np.array(
                [[0.539009, 0.579343, -0.118352], [0.082546, 0.866121, 0.051332], [-0.007136, -0.011959, 1.019095]]),
            5: np.array(
                [[0.458064, 0.679578, -0.137642], [0.092785, 0.846313, 0.060902], [-0.007494, -0.016807, 1.024301]]),
            6: np.array(
                [[0.385450, 0.769005, -0.154455], [0.100526, 0.829802, 0.069673], [-0.007442, -0.022190, 1.029632]]),
            7: np.array(
                [[0.319627, 0.849633, -0.169261], [0.106241, 0.815969, 0.077790], [-0.007025, -0.028051, 1.035076]]),
            8: np.array(
                [[0.259411, 0.923008, -0.182420], [0.110296, 0.804340, 0.085364], [-0.006276, -0.034346, 1.040622]]),
            9: np.array(
                [[0.203876, 0.990338, -0.194214], [0.112975, 0.794542, 0.092483], [-0.005222, -0.041043, 1.046265]]),
            10: np.array(
                [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]])
            # Add other cases for severity 2 to 10
        }
        return switch_severity.get(severity, np.zeros((3, 3)))

    elif deficiency == 'DEUTAN':
        switch_severity = {
            0: np.array(
                [[1.000000, 0.000000, -0.000000], [0.000000, 1.000000, 0.000000], [-0.000000, -0.000000, 1.000000]]),
            1: np.array(
                [[0.866435, 0.177704, -0.044139], [0.049567, 0.939063, 0.011370], [-0.003453, 0.007233, 0.996220]]),
            2: np.array(
                [[0.760729, 0.319078, -0.079807], [0.090568, 0.889315, 0.020117], [-0.006027, 0.013325, 0.992702]]),
            3: np.array(
                [[0.675425, 0.433850, -0.109275], [0.125303, 0.847755, 0.026942], [-0.007950, 0.018572, 0.989378]]),
            4: np.array(
                [[0.605511, 0.528560, -0.134071], [0.155318, 0.812366, 0.032316], [-0.009376, 0.023176, 0.986200]]),
            5: np.array(
                [[0.547494, 0.607765, -0.155259], [0.181692, 0.781742, 0.036566], [-0.010410, 0.027275, 0.983136]]),
            6: np.array(
                [[0.498864, 0.674741, -0.173604], [0.205199, 0.754872, 0.039929], [-0.011131, 0.030969, 0.980162]]),
            7: np.array(
                [[0.457771, 0.731899, -0.189670], [0.226409, 0.731012, 0.042579], [-0.011595, 0.034333, 0.977261]]),
            8: np.array(
                [[0.422823, 0.781057, -0.203881], [0.245752, 0.709602, 0.044646], [-0.011843, 0.037423, 0.974421]]),
            9: np.array(
                [[0.392952, 0.823610, -0.216562], [0.263559, 0.690210, 0.046232], [-0.011910, 0.040281, 0.971630]]),
            10: np.array(
                [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]])
        }
        return switch_severity.get(severity, np.zeros((3, 3)))

    elif deficiency == 'TRITAN':
        switch_severity = {
            0: np.array(
                [[1.000000, 0.000000, -0.000000], [0.000000, 1.000000, 0.000000], [-0.000000, -0.000000, 1.000000]]),
            1: np.array(
                [[0.926670, 0.092514, -0.019184], [0.021191, 0.964503, 0.014306], [0.008437, 0.054813, 0.936750]]),
            2: np.array(
                [[0.895720, 0.133330, -0.029050], [0.029997, 0.945400, 0.024603], [0.013027, 0.104707, 0.882266]]),
            3: np.array(
                [[0.905871, 0.127791, -0.033662], [0.026856, 0.941251, 0.031893], [0.013410, 0.148296, 0.838294]]),
            4: np.array(
                [[0.948035, 0.089490, -0.037526], [0.014364, 0.946792, 0.038844], [0.010853, 0.193991, 0.795156]]),
            5: np.array(
                [[1.017277, 0.027029, -0.044306], [-0.006113, 0.958479, 0.047634], [0.006379, 0.248708, 0.744913]]),
            6: np.array(
                [[1.104996, -0.046633, -0.058363], [-0.032137, 0.971635, 0.060503], [0.001336, 0.317922, 0.680742]]),
            7: np.array(
                [[1.193214, -0.109812, -0.083402], [-0.058496, 0.979410, 0.079086], [-0.002346, 0.403492, 0.598854]]),
            8: np.array(
                [[1.257728, -0.139648, -0.118081], [-0.078003, 0.975409, 0.102594], [-0.003316, 0.501214, 0.502102]]),
            9: np.array(
                [[1.278864, -0.125333, -0.153531], [-0.084748, 0.957674, 0.127074], [-0.000989, 0.601151, 0.399838]]),
            10: np.array(
                [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]])
        }
        return switch_severity.get(severity, np.zeros((3, 3)))


def apply_color_matrix(im, m):
    """Transform a color array with the given 3x3 matrix using PyTorch.

    Parameters
    ==========
    im : torch.Tensor of shape (..., 3)
        Input tensor with 3-channel color as the last dimension.

    m : torch.Tensor of shape (3, 3)
        Color matrix to apply.

    Returns
    =======
    im : torch.Tensor of shape (..., 3)
        Output tensor where each color vector is transformed by m.
    """
    return im @ m.T


def mysimColorBindImg(img, deficiency, severity):
    """
    Simulate color vision deficiency (CVD) for an image.

    Parameters
    ==========
    img : torch.Tensor of shape (B, C, H, W)
        Input image tensor.

    deficiency : str
        Type of color deficiency ('PROTAN', 'DEUTAN', 'TRITAN').

    severity : float
        Severity of the deficiency (0.0 to 1.0).

    Returns
    =======
    imgSim : torch.Tensor of shape (B, C, H, W)
        Simulated image tensor with the same shape as input.
    """
    # 计算 severity 的下限和上限
    severity_lower = int(severity * 10)
    severity_higher = min(severity_lower + 1, 10)
    # 获取颜色变换矩阵
    m1 = machado_2009_matrices(deficiency, severity_lower)
    m2 = machado_2009_matrices(deficiency, severity_higher)
    # 将 NumPy 数组转换为 PyTorch 张量
    m1 = torch.tensor(m1, dtype=img.dtype, device=img.device)
    m2 = torch.tensor(m2, dtype=img.dtype, device=img.device)
    # 插值计算最终矩阵
    alpha = severity * 10.0 - severity_lower
    m = alpha * m2 + (1.0 - alpha) * m1
    # 调整输入张量的形状为 (B, H, W, C)
    img = img.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
    # 应用颜色变换矩阵
    imgSim = apply_color_matrix(img, m)
    # 恢复原始形状 (B, C, H, W)
    imgSim = imgSim.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
    return imgSim

# def apply_color_matrix(im, m):
#     """Transform a color array with the given 3x3 matrix.
#
#     Parameters
#     ==========
#     im : array of shape (...,3)
#         Can be an image or a 1D array, as long as the last
#         dimension is a 3-channels color.
#
#     m : array of shape (3,3)
#         Color matrix to apply.
#
#     Returns
#     =======
#     im : array of shape (...,3)
#         Output array, where each input color vector was multiplied by m.
#     """
#     # Another option is np.einsum('ij, ...j', m, im), but it can be much
#     # slower, especially on on float32 types because the matrix multiplication
#     # is heavily optimized.
#     # So the matmul is generally (much) faster, but we need to take the
#     # transpose of m as it gets applied on the right side. Indeed for each
#     # column color vector v we wanted $v' = m . v$ . To flip the side we can
#     # use $m . v = (v^T . m^T)^T$ . The transposes on the 1d vector are implicit
#     # and can be ignored, so we just need to compute $v . m^T$. This is what
#     # numpy matmul will do for all the vectors thanks to its broadcasting rules
#     # that pick the last 2 dimensions of each array, so it will actually compute
#     # matrix multiplications of shape (M,3) x (3,3) with M the penultimate dimension
#     # of m. That will write a matrix of shape (M,3) with each row storing the
#     # result of $v' = v . M^T$.
#     return im @ m.T

# def mysimColorBindImg(img, deficiency, severity): #输入是(B,C,H,W) 输出是(B,C,H,W)
#     # print(img.shape)
#     severity_lower = float(severity * 10.0)
#     severity_higher = min(severity_lower + 1, 10)
#     m1 = machado_2009_matrices(deficiency, severity_lower)
#     m2 = machado_2009_matrices(deficiency, severity_higher)
#     alpha = (severity - severity_lower / 10.0)
#     m = alpha * m2 + (1.0 - alpha) * m1
#     img = img.cpu().detach().numpy()
#     img = np.transpose(img, (0, 2, 3, 1))
#     imgSim = apply_color_matrix(img,m)
#     # print(imgSim.shape) #(4, 128, 128, 3)
#     imgSim = np.transpose(imgSim, (0, 3, 1, 2))
#     imgSim= torch.tensor(imgSim)
#     imgSim = imgSim.to('cuda')
#     imgSim = imgSim.float()
#     # print(imgSim.shape) #(4, 3, 128, 128)
#     return imgSim

def mysimColorBindImg_val(img, deficiency, severity): #输入是(C,H,W)输出是(H,W,C)
    severity_lower = float(severity * 10.0)
    severity_higher = min(severity_lower + 1, 10)
    m1 = machado_2009_matrices(deficiency, severity_lower)
    m2 = machado_2009_matrices(deficiency, severity_higher)
    alpha = (severity - severity_lower / 10.0)
    m = alpha * m2 + (1.0 - alpha) * m1
    img = img.permute(1, 2, 0)
    imgSim = apply_color_matrix(img,m)
    imgSim= torch.tensor(imgSim)
    imgSim = imgSim.to('cuda')
    imgSim = imgSim.float()
    # print(imgSim.shape) #(4, 3, 128, 128)
    return imgSim



