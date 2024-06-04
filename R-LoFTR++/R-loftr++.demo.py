import torch
import cv2
import numpy as np
import matplotlib.cm as cm

from src.utils.plotting import make_matching_figure
from src.loftr import LoFTR, default_cfg


if __name__ == '__main__':
    #预训练模型选择
    image_type = ''
	# 根据个人图片进行修改
    img0_pth = r'image0/pth'
    img1_pth = r'image1/pth'
    image_pair = [img0_pth, img1_pth]

    # The default config uses dual-softmax.
    # The outdoor and indoor models share the same config.
    # You can change the default values like thr and coarse_match_type.
    matcher = LoFTR(config=default_cfg)
    # load中修改预训练模型地址
    if image_type == 'indoor':
        matcher.load_state_dict(torch.load("")['state_dict'])
    elif image_type == 'outdoor':
        matcher.load_state_dict(torch.load("")['state_dict'])
    else:
        raise ValueError("Wrong image_type is given.")
    matcher = matcher.eval().cuda()

    # Rerun this cell (and below) if a new image pair is uploaded.
    img0_raw = cv2.imread(image_pair[0], cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(image_pair[1], cv2.IMREAD_GRAYSCALE)
    img0_raw = cv2.resize(img0_raw, (640, 480))
    img1_raw = cv2.resize(img1_raw, (640, 480))

    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
    batch = {'image0': img0, 'image1': img1}

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()

    # Draw
    color = cm.jet(mconf, alpha=0.7)
    text = [
        'R-LoFTR++'.format(len(mkpts0)),
    ]
    fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, mkpts0, mkpts1)

    # A high-res PDF will also be downloaded automatically.
    make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, mkpts0, mkpts1,  path=r'')