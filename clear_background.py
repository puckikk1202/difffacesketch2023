from PIL import Image
import numpy as np
import os
from tqdm import tqdm

NAME_LIST = ['00', '_skin.png', '_nose.png', '_eye_g.png', '_l_eye.png', '_r_eye.png', '_l_brow.png', '_r_brow.png', '_l_ear.png', '_r_ear.png', '_mouth.png', '_u_lip.png', '_l_lip.png', '_hair.png', '_hat.png', '_ear_r.png', '_neck_l.png', '_neck.png', '_cloth.png']
img_path = './datasets/celeba_img/'
mask_path = './datasets/celeba_mask/'
save_path = './datasets/celeba_img_nobg/'

for img in tqdm(os.listdir(img_path)):
    pil_img = np.asarray(Image.open(os.path.join(img_path, img)).resize((512, 512), Image.LANCZOS))
    mask_array = np.zeros((512, 512))
    for mask_name in NAME_LIST:
        mask_name = mask_path + img.split('.')[0].zfill(5) + mask_name
        if os.path.isfile(mask_name):
            mask_img = np.asarray(Image.open(mask_name).convert('L'))
            mask_array += mask_img
    # print(np.max(mask_array)) 
    mask_bool = mask_array > 1
    hoge = np.ones((512, 512))
    bg = mask_bool * hoge
    # print(mask)
    for y, yy in enumerate(bg):
        for x, xx in enumerate(yy):
            if xx== 0:
                pil_img[y][x] = np.full(3, 255)
    # clear_array = pil_img * bg
    # print(np.max(pil_img))
    clear_img = Image.fromarray(pil_img.astype(np.uint8))
    clear_img.save(save_path + img)
