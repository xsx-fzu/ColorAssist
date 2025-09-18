import torch
import core.metrics as Metrics
import numpy as np

def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func

# def get_cand_err2(model, cand, data, args):
#     avg_psnr = 0.0
#     idx = 0
#     for _,  val_data in enumerate(data):
#         idx += 1
#         model.feed_data(val_data)
#         model.test(cand=cand, continous=True)
#
#         visuals = model.get_current_visuals()
#
#         hr_img = Metrics.tensor2img(visuals['original'])  # uint8
#         sr_img = Metrics.tensor2img(visuals['recolor'][-1])
#         psnr = Metrics.calculate_psnr(sr_img, hr_img)
#         avg_psnr += psnr
#     avg_psnr = avg_psnr / idx
#     return avg_psnr

def get_cand_err2(model, cand, data, args):
    psnr_list = []
    for _, val_data in enumerate(data):
        model.feed_data(val_data)
        model.test(cand=cand, continous=True)

        visuals = model.get_current_visuals()

        hr_img = Metrics.tensor2img(visuals['original'])
        sr_img = Metrics.tensor2img(visuals['recolor'][-1])
        psnr = Metrics.calculate_psnr(sr_img, hr_img)

        # 排除 PSNR 为无穷大的异常值
        if not np.isinf(psnr):
            psnr_list.append(psnr)

    # 计算平均 PSNR
    if psnr_list:
        avg_psnr = np.mean(psnr_list)
    else:
        avg_psnr = 0.0

    return avg_psnr
