# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def crop(img_file, mask_file):
    # name, *_ = img_file.split(".")
    img_array = np.array(Image.open(img_file))
    mask = np.array(Image.open(mask_file))

    #通过将原图和mask图片归一化值相乘，把背景转成黑色
    #从mask中随便找一个通道，cat到RGB后面，最后转成RGBA
    # res = np.concatenate((img_array * (mask/255), mask[:, :, [0]]), -1)
    # print(res.shape)
    res = np.concatenate((img_array, mask[:, :, [0]]), -1)
    img = Image.fromarray(res.astype('uint8'), mode='RGBA')
    # img.show()
    return img


if __name__ == "__main__":
    import os

    model = "u2net"
    # model = "u2netp"

    img_root = "test_data/test_images"
    mask_root = "test_data/{}_results".format(model)
    crop_root = "test_data/{}_crops".format(model)
    os.makedirs(crop_root, mode=0o775, exist_ok=True)

    for img_file in os.listdir(img_root):
        print("crop image {}".format(img_file))
        name, *_ = img_file.split(".")
        res = crop(
            os.path.join(img_root,  img_file),
            os.path.join(mask_root, name + ".png")
        )
        res.save(os.path.join(crop_root, name + "_crop.png"))
        # exit()
