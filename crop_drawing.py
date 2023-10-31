from PIL import Image
import numpy as np
import os

# path_512 = './datasets/celeba_sketch_nobg_512/'
# path_256 = './datasets/celeba_sketch_nobg_256/'
# path_128 = './datasets/celeba_sketch_nobg_128/'

def SRA_augmenttation(src_512=None, src_256=None, src_128=None, data_dir=None):

    canvas_128 = np.zeros((5, 512, 512))
    canvas_256 = np.zeros((5, 512, 512))
    canvas_512 = np.zeros((5, 512, 512))
    save_path = './fig/'

    img_128 = Image.open(src_128).convert('L').resize((512, 512), Image.LANCZOS)
    img_128 = np.array(img_128)
    img_256 = Image.open(src_256).convert('L').resize((512, 512), Image.LANCZOS)
    img_256 = np.array(img_256)
    img_512 = Image.open(src_512).convert('L').resize((512, 512), Image.LANCZOS)
    img_512 = np.array(img_512)

    for i in range(5):
        for y, _ in enumerate(img_128):
            for x, v in enumerate(img_128):
                if i == 0:
                    if x > 108 and x < 108+128 and y > 188 and y < 188+128:
                        canvas_128[i][y][x] = img_128[y][x]
                elif i == 1:
                    if x > 256 and x < 256+128 and y > 188 and y < 188+128:
                        canvas_128[i][y][x] = img_128[y][x]
                elif i == 2:
                    if x > 182 and x < 182+192 and y > 232 and y < 232+192:
                        canvas_128[i][y][x] = img_128[y][x]
                elif i == 3:
                    if x > 170 and x < 170+192 and y > 302 and y < 302+192:
                        canvas_128[i][y][x] = img_128[y][x]
                else:
                    if not (x > 108 and x < 108+128 and y > 188 and y < 188+128) and not (x > 256 and x < 256+128 and y > 188 and y < 188+128) and not (x > 182 and x < 182+192 and y > 232 and y < 232+192) and not (x > 170 and x < 170+192 and y > 302 and y < 302+192):
                        canvas_128[i][y][x] = img_128[y][x]

    for i in range(5):
        for y, _ in enumerate(img_256):
            for x, v in enumerate(img_256):
                if i == 0:
                    if x > 108 and x < 108+128 and y > 188 and y < 188+128:
                        canvas_256[i][y][x] = img_256[y][x]
                elif i == 1:
                    if x > 256 and x < 256+128 and y > 188 and y < 188+128:
                        canvas_256[i][y][x] = img_256[y][x]
                elif i == 2:
                    if x > 182 and x < 182+192 and y > 232 and y < 232+192:
                        canvas_256[i][y][x] = img_256[y][x]
                elif i == 3:
                    if x > 170 and x < 170+192 and y > 302 and y < 302+192:
                        canvas_256[i][y][x] = img_256[y][x]
                else:
                    if not (x > 108 and x < 108+128 and y > 188 and y < 188+128) and not (x > 256 and x < 256+128 and y > 188 and y < 188+128) and not (x > 182 and x < 182+192 and y > 232 and y < 232+192) and not (x > 170 and x < 170+192 and y > 302 and y < 302+192):
                        canvas_256[i][y][x] = img_256[y][x]

    for i in range(5):
        for y, _ in enumerate(img_512):
            for x, v in enumerate(img_512):
                if i == 0:
                    if x > 108 and x < 108+128 and y > 188 and y < 188+128:
                        canvas_512[i][y][x] = img_512[y][x]
                elif i == 1:
                    if x > 256 and x < 256+128 and y > 188 and y < 188+128:
                        canvas_512[i][y][x] = img_512[y][x]
                elif i == 2:
                    if x > 182 and x < 182+192 and y > 232 and y < 232+192:
                        canvas_512[i][y][x] = img_512[y][x]
                elif i == 3:
                    if x > 170 and x < 170+192 and y > 302 and y < 302+192:
                        canvas_512[i][y][x] = img_512[y][x]
                else:
                    if not (x > 108 and x < 108+128 and y > 188 and y < 188+128) and not (x > 256 and x < 256+128 and y > 188 and y < 188+128) and not (x > 182 and x < 182+192 and y > 232 and y < 232+192) and not (x > 170 and x < 170+192 and y > 302 and y < 302+192):
                        canvas_512[i][y][x] = img_512[y][x]

    for n in range(10):
        img = np.zeros((512, 512))
        ran = np.random.randint(3, size=5)
        for i in range(5):
            print(ran[i])
            
            if ran[i] == 0:
                img = img + canvas_128[4-i]
            if ran[i] == 1:
                img = img + canvas_256[4-i] 
            if ran[i] == 2:
                img = img + canvas_512[4-i]

        img = Image.fromarray(img).convert('L')
        # img.show()
        img = img.save(data_dir+str(n)+'.jpg')

# img1 = Image.fromarray(canvas_256[0])
# img1.show()
# img2 = Image.fromarray(canvas_256[1])
# img2.show()
# img3 = Image.fromarray(canvas_256[2])
# img3.show()
# img4 = Image.fromarray(canvas_256[3])
# img4.show()
# img5 = Image.fromarray(canvas_256[4])
# img5.show()

if __name__ == '__main__':
    path_512 = './datasets/celeba_sketch_nobg_512/'
    path_256 = './datasets/celeba_sketch_nobg_256/'
    path_128 = './datasets/celeba_sketch_nobg_128/'
    data_dir = './datasets/augmented_sketch/'
    for f in os.listdir(path_512):
        src_512 = os.path.join(path_512,f)
        src_256 = os.path.join(path_256,f)
        src_128 = os.path.join(path_128,f)
        SRA_augmenttation(src_512, src_256, src_128, data_dir)