import numpy as np
import torch

from PIL import Image

from LPRNet import LPRNet

import warnings

from dataset import CVLicensePlateDataset

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    ds = CVLicensePlateDataset('data', subset='train', download=False)

    target_label = 'AB8360DT'
    img_path = f'data/CVLicensePlateDataset/raw/cvlpr_cropped_test_20240418/AB8360DT_12.jpeg'
    ckpt_path = 'ckpt/20240616/cvlpr_ckpt_1000.pth'

    model = LPRNet(num_classes=37)

    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    img = torch.Tensor(np.asarray(Image.open(img_path)))
    img = img.permute((2, 0, 1))
    img = img.unsqueeze(0)

    logits = model(img)
    mean_logits = logits.mean(dim=2).detach()

    print(mean_logits.shape)

    result = mean_logits.squeeze(0).permute((1, 0)).cpu().numpy()

    result_array = []
    for prob in result:
        result_array.append(prob.argmax())

    converted_to_label = [ds.labels_dict[idx] for idx in result_array]
    print(converted_to_label)
