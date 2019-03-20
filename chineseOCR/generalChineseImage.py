from PIL import Image, ImageDraw, ImageFont
import codecs as cs
import numpy as np
import random
import os
import skimage
import cv2

from skimage import morphology
from skimage.morphology import square
import skimage.filters

#from image_helper import *



import numpy as np
from PIL import Image

# const value
normal_height=64
normal_width=64

def get_binary_image(img,maxVal=255):
    bin_img=None
    if maxVal==255:
        _,bin_img=cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    elif maxVal==0:
        _,bin_img=cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return bin_img
def get_dilate_image(img,stride=10):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (stride, stride))
    dilate_image = cv2.dilate(img, kernel)
    return dilate_image

def cv2_resize_image(img,width):
    old_height,old_width=img.shape[:2]
    ratio=float(old_height)/old_width
    height=int(width*ratio)
    resize_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    return resize_img

def draw_block(img,block, color= (0,0,255)):
    if block.get_block_type()=="char":
        cv2.rectangle(img, ( block.top_x, block.top_y), (block.bottom_x, block.bottom_y), color, thickness=1)
    if block.get_block_type()=="punctuation":
        cv2.rectangle(img, ( block.top_x, block.top_y), (block.bottom_x, block.bottom_y), (0,255,0), thickness=1)
    for v in block.child_blocks:
        img=draw_block(img, v, color)
    return img

def show_image(img,text="img"):
    #cv2.imshow(text, img)
    cv2.imshow(text, np.array(img,dtype=np.uint8))
    cv2.waitKey()


def get_char_area(char_img):
    #print charImg.shape
    binArr = char_img/255
    rowSum = np.sum(binArr, axis=1)
    colSum = np.sum(binArr, axis=0)
    height,width = char_img.shape[:2]
    left = 0
    for i,val in enumerate(colSum):
        if val!=0:
            left = i
            break
    right = width-1
    for i,val in enumerate(colSum[::-1]):
        if val!=0:
            right = width - 1 - i
            break
    top = 0
    for i,val in enumerate(rowSum):
        if val!=0:
            top = i
            break
    bottom = height-1
    for i,val in enumerate(rowSum[::-1]):
        if val!=0:
            bottom = height-1-i
            break
    #print top,bottom,left,right
    return char_img[top:bottom+1,left:right+1]

def get_char_area2(char_img):
    #print charImg.shape
    binArr = char_img/255
    r,c=char_img.shape
    rowSum = np.sum(binArr, axis=1)
    colSum = np.sum(binArr, axis=0)
    height,width = char_img.shape[:2]
    left = 0
    for i,val in enumerate(colSum):
        if val<r:
            left = i
            break
    right = width-1
    for i,val in enumerate(colSum[::-1]):
        if val<r:
            right = width - 1 - i
            break
    top = 0
    for i,val in enumerate(rowSum):
        if val<c:
            top = i
            break
    bottom = height-1
    for i,val in enumerate(rowSum[::-1]):
        if val<c:
            bottom = height-1-i
            break
    return char_img[top:bottom+1,left:right+1]


def padding_image(src_img):
    img = get_char_area(src_img)
    #cv2.imshow('padding_image2', img)
    #cv2.waitKey(0)
    img = 255 - img
    height,width = img.shape[:2]
    if height > width:
        lpad = (height-width)/2
        rpad = (height-width) - lpad
        newImg = Image.new("L", (height, height), 255)
        newImg = np.array(newImg)
        newImg[0:height,(lpad+1):(height-rpad+1)] = img
    elif height < width:
        tpad = (width-height)/2
        bpad = (width-height) - tpad
        newImg = Image.new("L", (width, width), 255)
        newImg = np.array(newImg)
        newImg[tpad+1:(width-bpad+1),0:width] = img
    else:
        newImg = img
    return newImg

def resize_char_image(img, width, scale=1):
    newimg = padding_image(img)
    h,w = newimg.shape[:2]
    if h!=w:
        exit()
    if scale==1:
        newWidth = width
    else:
        newWidth = int(w*scale)
    resize_img = cv2.resize(newimg, (newWidth, newWidth), interpolation =cv2.INTER_CUBIC)
    normImg = Image.new("L", (normal_height, normal_width), 255)
    normImg = np.array(normImg)
    lpad = (normal_width-newWidth)/2
    rpad = normal_width - newWidth - lpad
    normImg[lpad:normal_height-rpad, lpad:normal_width-rpad] = resize_img
    return normImg

def resize_char_image2(img, num_height=64, num_width=64):
    #print np.array(img)
    img = get_char_area(img)
    #print np.array(img)
    img = 255 - img

    h, w=img.shape[:2]
    new_img=np.zeros((num_height,num_width),dtype=int)+255
    pad_h=(num_height-h)/2
    pad_w=(num_width-w)/2
    if len(img.shape)==3:
        new_img[pad_h:h+pad_h, pad_w:w+pad_w]=img[:,:,0]
    if len(img.shape)==2:
        new_img[pad_h:h+pad_h, pad_w:w+pad_w]=img[:,:]
    #print new_img
    new_img=np.array(new_img,dtype=np.uint8)
    return new_img

def img_to_file(base64str,file_name):
    file = open(file_name, 'wb')
    file.write(base64str)
    file.close()

#open('a.png', 'r').read() to numpy.ndarray, such as cv2.imread('a.png', 1)
def raw_data_to_img_ndarray(imgdata):
    file_bytes = np.asarray(bytearray(imgdata), dtype=np.uint8)
    img_data_ndarray = cv2.imdecode(file_bytes, 1)
    return img_data_ndarray





def genFontImage(font, char, color=255):
    size = font.size
    image = Image.new('L', (size, size), color=color)
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), char, font=font, fill='#000000')
    # print np.array(image)
    return image


# 字体识别网站
# http://www.likefont.com/

# 48 50 57

my_random = random.Random()
my_random.seed()


def get_random(s, e):
    return my_random.randint(s, e)


def test_image_font():
    size = 20
    # font = ImageFont.truetype('/Library/Fonts/华文仿宋.ttf', size)
    # font = ImageFont.truetype('train/fireflysung.ttf', size)
    font = ImageFont.truetype('train/font_test/方正瘦金书_GBK.ttf', size)
    han = u'安'
    # tmp_img=gen_font_image2(font, han)

    image = genFontImage(font, han)
    image = np.array(image)
    show_image(image)
    image = get_char_area2(image)
    show_image(image)
    image = format_train_image_random(image)
    image = image[:, :, 0]
    print ("aaaa", image)
    tmp_value = skimage.util.random_noise(image, mean=1.0)
    image = tmp_value * image
    tmp_img = np.array(image, dtype=int)
    print (tmp_img)
    """
    image = genFontImage(font,han)
    tmp_value=skimage.util.random_noise(np.array(image))
    tmp_img=tmp_value*np.array(image)
    """
    # print tmp_img
    cv2.imwrite('t2.png', tmp_img)
    # image.save('tt.png')


def load_data(img_dir):
    img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    img_list = []
    label_list = []
    for v in img_files:
        file_name = os.path.join(img_dir, v)
        src_img = cv2.imread(file_name, 1)
        tmp_img = format_train_image(src_img)
        img_list.append(tmp_img)
        label = v.split('_')[0]
        label_list.append(int(label))
        print (label)
        # print tmp_img[20:25,0:30]
    return img_list, label_list


def format_train_image(img, num_height=64, num_width=64):
    h, w = img.shape[:2]
    new_img = np.zeros((num_height, num_width, 1), dtype=float) + 255.0
    pad_h = (num_height - h) / 2
    pad_w = (num_width - w) / 2
    if len(img.shape) == 3:
        new_img[pad_h:h + pad_h, pad_w:w + pad_w, 0] = img[:, :, 0]
    if len(img.shape) == 2:
        new_img[pad_h:h + pad_h, pad_w:w + pad_w, 0] = img[:, :]
    return new_img


def format_train_image_random(img, num_height=64, num_width=64):
    h, w = img.shape[:2]
    new_img = np.zeros((num_height, num_width, 1), dtype=int) + 255
    pad_h = (num_height - h) / 2
    pad_w = (num_width - w) / 2
    pad_h = get_random(pad_h - 5, pad_h + 5)
    pad_w = get_random(pad_w - 5, pad_w + 5)
    if len(img.shape) == 3:
        new_img[pad_h:h + pad_h, pad_w:w + pad_w, 0] = img[:, :, 0]
    if len(img.shape) == 2:
        new_img[pad_h:h + pad_h, pad_w:w + pad_w, 0] = img[:, :]
    return new_img


def format_train_image_random2(img, num_height=64, num_width=64):
    h, w = img.shape[:2]
    new_img = np.zeros((num_height, num_width, 1), dtype=np.uint8) + 255
    pad_h = int((num_height - h) / 2)
    pad_w = int((num_width - w) / 2)
    pad_h = get_random(pad_h - 5, pad_h + 5)
    pad_w = get_random(pad_w - 5, pad_w + 5)
    if len(img.shape) == 3:
        new_img[pad_h:h + pad_h, pad_w:w + pad_w, 0] = img[:, :, 0]
    if len(img.shape) == 2:
        new_img[pad_h:h + pad_h, pad_w:w + pad_w, 0] = img[:, :]
    return new_img


def load_char_table(file):
    # file='train/hanzi_table.txt'
    examples = list(open(file, "r").readlines())
    hanzi_dict = {}
    for v in examples:
        arr = v.strip().split(' ')
        if len(arr) < 2:
            continue
        hanzi_dict[arr[0]] = arr[1]
    return hanzi_dict


dict_list = ['1353', '1336', '433', '2597', '1273', '2848', '796', '1755']


def create_char_sample_test_mini():
    file = 'train/hanzi_table.txt'
    hanzi_dict = load_char_table(file)
    sample_dir = "train/samples_test_mini"
    font_file = 'train/font_test/华文仿宋.ttf'

    size = 30
    font = ImageFont.truetype(font_file, size)
    i = 0
    while i < 1000:
        k = dict_list[i % len(dict_list)]
        v = hanzi_dict[k]
        tmp_img = gen_font_image2(font, v, "noise")
        if tmp_img is None:
            print (v, " : can not create.")
            continue
        # tmp_img=format_train_image(tmp_img)
        file_name = os.path.join(sample_dir, '%s_%d_%d.png' % (k, size, get_random(1, 1000)))
        cv2.imwrite(file_name, tmp_img)
        i = i + 1


def bmp_to_png():
    img_dir = 'train/samples2'
    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    for v in img_files:
        print (v)
        color_img = cv2.imread(v)
        v = v.replace(".bmp", '.png')
        v = v.replace("samples2", 'samples3')
        cv2.imwrite(v, color_img)


def test_erosion():
    sample_dir = "test/samples"
    font_file = 'train/font/fireflysung.ttf'
    # font_file='train/font/方正粗圆简体.ttf'

    size = 30
    font = ImageFont.truetype(font_file, size)
    image = gen_font_image(font, u'位', interfer_type='')
    image = np.array(image, dtype=np.uint8)
    cv2.imshow('src', image)
    cv2.waitKey()

    bin_img = get_binary_image(image)
    kernel = np.ones((1, 1), np.uint8)  # 生成一个1x1的核
    # kernel=np.array([1,0],np.uint8)
    erosion = cv2.erode(bin_img, kernel, iterations=1)  # 调用腐蚀算法
    cv2.imshow('erosion', 255 - erosion)
    cv2.waitKey()

    kernel = np.array([[3, 3]], np.uint8)
    d1 = cv2.dilate(bin_img, kernel, iterations=1)
    cv2.imshow('erosion11', 255 - d1)
    cv2.waitKey()
    kernel = np.ones((2, 2), np.uint8)
    # kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(1,2))
    erosion = cv2.erode(d1, kernel, iterations=1)  # 调用腐蚀算法
    cv2.imshow('erosion1', 255 - erosion)
    cv2.waitKey()

    # e2=skimage.morphology.dilation(bin_img)
    e2 = skimage.morphology.erosion(image)
    cv2.imshow('erosion2', e2)
    cv2.waitKey()

    cv2.imshow("noise", np.array(bin_img, dtype=np.uint8))
    cv2.waitKey()


def test_erosion2():
    sample_dir = "test/samples"
    # font_file='train/font/fireflysung.ttf'
    font_file = 'train/font/方正粗圆简体.ttf'
    # font_file='train/font/方正粗宋_GBK.ttf'
    # font_file='train/font/方正水柱_GBK.ttf'
    size = 35
    font = ImageFont.truetype(font_file, size)
    char = u'豪'
    # 细字体 >30
    # ['','noise','filter','closing']
    # 粗圆字体 >30
    # ['','noise','filter','opening']
    image_list = gen_font_image4(font, char, ['noise', 'filter', 'erosion', 'opening', 'closing'])
    for v in image_list:
        cv2.imshow("noise", np.array(v, dtype=np.uint8))
        cv2.waitKey()


def gen_font_image(font, char, interfer_type="noise"):
    image = genFontImage(font, char)
    a = np.array(image)
    # 字体无字
    if a[np.where(a < 50)].size < 5:
        return None

    if interfer_type == "noise":
        # 高斯模糊
        tmp_value = skimage.util.random_noise(np.array(image))
        tmp_img = tmp_value * np.array(image)
        return tmp_img
    if interfer_type == "erosion":
        # 腐蚀
        image = skimage.morphology.erosion(np.array(image))
        return image
    if interfer_type == "filter":
        # 滤波
        val = skimage.filters.gaussian(np.array(image), sigma=5)
        image = np.array(image) * val
        return image
    return np.array(image)


def gen_font_image2(font, char, interfer_type="noise"):
    image = genFontImage(font, char)
    image = np.array(image)
    # 字体无字
    if image[np.where(image < 100)].size < 4:
        # print image
        # show_image(image)
        return None
    image = get_char_area2(image)
    if interfer_type == "noise":
        # 高斯模糊
        image = format_train_image_random(image)
        # print "aaaa",image
        tmp_value = skimage.util.random_noise(image, mean=1.0)
        image = tmp_value * image
        image = np.array(image, dtype=int)
        # print tmp_value
        # print image
        # exit(0)
        return image
    if interfer_type == "erosion":
        # 腐蚀
        image = skimage.morphology.erosion(image)
        # 先膨胀再腐蚀(闭运算)
        # image=get_binary_image(image)
        # image=skimage.morphology.dilation(image)
        # image=skimage.morphology.erosion(image, skimage.morphology.disk(1))
        # image=255-image
    if interfer_type == "filter":
        # 滤波
        val = skimage.filters.gaussian(image, sigma=1)
        image = image * val
    image = format_train_image_random(image)
    return image


def gen_font_image3(font, char, interfer_type="noise"):
    image = genFontImage(font, char)
    image = np.array(image)
    # 字体无字
    if image[np.where(image < 100)].size < 4:
        # print image
        # show_image(image)
        return None
    image = get_char_area2(image)
    if interfer_type == "noise":
        # 高斯模糊
        image = format_train_image_random(image)
        # print "aaaa",image
        tmp_value = skimage.util.random_noise(image, mean=1.0)
        image = tmp_value * image
        image = np.array(image, dtype=int)
        # print tmp_value
        # print image
        # exit(0)
        return image
    if interfer_type == "erosion":
        # 腐蚀
        # image=skimage.morphology.erosion(image)
        # 先膨胀再腐蚀(闭运算)
        image = get_binary_image(image)
        # image=skimage.morphology.dilation(image)
        image = skimage.morphology.erosion(image, skimage.morphology.disk(1))
        image = 255 - image
    if interfer_type == "filter":
        # 滤波
        val = skimage.filters.gaussian(image, sigma=1)
        image = image * val
    image = format_train_image_random(image)
    return image


def gen_font_image4(font, char, interfer_type_list=[""]):
    image = genFontImage(font, char)
    image = np.array(image)
    # 字体无字
    if image[np.where(image < 100)].size < 4:
        # print image
        # show_image(image)
        return None
    image = get_char_area2(image)
    tmp_image = format_train_image_random2(image)
    img_list = []
    for interfer_type in interfer_type_list:
        if interfer_type == "noise":
            # 噪声
            image = np.array(tmp_image, dtype=int)
            tmp_value = skimage.util.random_noise(image, mean=1.0)
            image = tmp_value * image
            img_list.append(image)
        elif interfer_type == "closing":
            # 先膨胀再腐蚀(闭运算)
            image = get_binary_image(tmp_image)
            image = skimage.morphology.dilation(image)
            image = skimage.morphology.erosion(image, skimage.morphology.disk(1))
            image = 255 - image
            img_list.append(image)
        elif interfer_type == "opening":
            # 先腐蚀再膨胀(开运算)
            image = get_binary_image(tmp_image)
            image = skimage.morphology.erosion(image, skimage.morphology.disk(1))
            image = skimage.morphology.dilation(image)
            image = 255 - image
            img_list.append(image)
        elif interfer_type == "erosion":
            # 腐蚀
            image = get_binary_image(tmp_image)
            image = skimage.morphology.erosion(image, skimage.morphology.disk(1))
            # image=skimage.morphology.dilation(image)
            image = 255 - image
            img_list.append(image)
        elif interfer_type == "filter":
            # 滤波
            image = get_binary_image(tmp_image)
            val = skimage.filters.gaussian(image, sigma=1)
            image = image * val
            image = 255 - image
            img_list.append(image)
        else:
            img_list.append(tmp_image)
    return img_list


def gen_font_image5(font, char, interfer_type_list=[""]):
    image = genFontImage(font, char)
    image = np.array(image)
    # 字体无字
    if image[np.where(image < 100)].size < 4:
        # print image
        # show_image(image)
        return None
    image = get_char_area2(image)
    tmp_image = format_train_image_random2(image)
    img_list = []
    for interfer_type in interfer_type_list:
        if interfer_type == "noise":
            # 噪声
            image = np.array(tmp_image, dtype=int)
            tmp_value = skimage.util.random_noise(image, mean=1.0)
            image = tmp_value * image
            img_list.append(image)
        elif interfer_type == "erosion":
            # 腐蚀
            image = skimage.morphology.erosion(tmp_image)
            # 先膨胀再腐蚀(闭运算)
            """
            image=get_binary_image(tmp_image)
            image=skimage.morphology.dilation(image)
            image=skimage.morphology.erosion(image, skimage.morphology.disk(1))
            image=255-image
            """
            img_list.append(image)
        elif interfer_type == "filter":
            # 滤波
            val = skimage.filters.gaussian(tmp_image, sigma=1)
            image = image * val
            img_list.append(image)
        elif interfer_type == "erosion2":
            image = get_binary_image(tmp_image)
            kernel = np.ones((1, 1), np.uint8)
            image = cv2.erode(image, kernel, iterations=1)  # 腐蚀
            img_list.append(255 - image)
        else:
            img_list.append(tmp_image)
    return img_list


def my_mkdir(dir_name):
    try:
        if os.path.exists(dir_name) == False:
            os.makedirs(dir_name)
    except:
        print ("mkdir error:%s" % dir_name)
        return False
    return True


def create_char_sample_test():
    file = 'train/hanzi_table.txt'
    hanzi_dict = load_char_table(file)
    sample_dir = "train/samples_test"
    font_file = 'train/font_test/华文仿宋.ttf'
    size = 30
    font = ImageFont.truetype(font_file, size)

    disturb_list = ['noise', 'erosion', 'filter']
    font_dir = 'train/font_test'
    font_files = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f))]
    font_obj_list = []
    for v in font_files:
        font = ImageFont.truetype(v, size)
        font_obj_list.append(font)

    i = 0
    for font in font_obj_list:
        for k, v in hanzi_dict.items():
            key_inx = get_random(0, len(hanzi_dict))
            if key_inx >= len(hanzi_dict):
                continue
            k = str(key_inx)
            v = hanzi_dict[k]
            tmp_img = gen_font_image2(font, v, "noise")
            if tmp_img is None:
                print (v, " : can not create.")
                continue
            file_name = os.path.join(sample_dir, '%s_%d_%d.png' % (k, size, get_random(1, 10000)))
            cv2.imwrite(file_name, tmp_img)
            i = i + 1


# 生成样本 方案1 test 0.97
def create_char_sample_train():
    sample_dir = 'train/samples4'
    char_file = 'train/hanzi_table.txt'
    hanzi_dict = load_char_table(char_file)

    font_dir = 'train/font'
    font_files = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f))]
    font_obj_list = []

    disturb_list = ['noise', 'erosion', 'filter']
    # char_size=[30,40,50]
    # biaodian_size=[15,20,25,30]
    # char_size=[25, 30, 35, 40, 45, 50]
    char_size = [15, 20, 25, 30, 35, 40, 45, 50]
    # char_size=[15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 50]
    font_obj_list = []
    for size in char_size:
        tmp_list = []
        for f in font_files:
            font = ImageFont.truetype(f, size)
            tmp_list.append(font)
        font_obj_list.append(tmp_list)

    eproch = 0
    while (eproch < 150):
        print ("eproch", eproch)
        for k, v in hanzi_dict.items():
            for interfer_type in disturb_list:
                font = None
                # 标点符号 数字 字体大小选为 15 ... 30
                # 字母 汉字 21 ... 48
                if int(k) < 20:
                    font = font_obj_list[get_random(0, 3)][get_random(0, len(font_files) - 1)]
                else:
                    font = font_obj_list[get_random(2, len(font_obj_list) - 1)][get_random(0, len(font_files) - 1)]
                image = gen_font_image2(font, v, interfer_type)
                if image is None:
                    print (v, " : can not create. font=", font.path)
                    continue

                font_name_size = font.path.replace('/', '_').replace('.', '_') + '_' + str(font.size)
                # save_dir=os.path.join(sample_dir, save_dir)
                save_dir = os.path.join(sample_dir, k)
                my_mkdir(save_dir)
                file_name = os.path.join(save_dir, '%s_%s_%d_%d.png' % (
                k, font_name_size, get_random(1, 10000), get_random(1, 10000)))
                cv2.imwrite(file_name, image)
        eproch = eproch + 1


# best runs/1505098258/checkpoints/
def create_char_sample_train3():
    sample_dir = 'train/samples4'
    char_file = 'train/hanzi_table.txt'
    hanzi_dict = load_char_table(char_file)

    font_dir = 'train/font'
    font_files = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f))]
    font_obj_list = []

    disturb_list = ['noise', 'filter', 'erosion', 'erosion2']
    # char_size=[30,40,50]
    # biaodian_size=[15,20,25,30]
    # char_size=[25, 30, 35, 40, 45, 50]
    char_size = [15, 20, 25, 30, 35, 45, 50]
    # char_size=[15, 20, 25, 30, 40, 50]
    # char_size=[15, 20, 25, 30, 35]
    # char_size=[15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 50]
    font_obj_list = []
    for size in char_size:
        for f in font_files:
            font = ImageFont.truetype(f, size)
            font_obj_list.append(font)

    for k, v in hanzi_dict.items():
        print (k)
        for font in font_obj_list:
            # 标点符号 数字 字体大小选为 15 ... 30
            if int(k) < 20 and font.size > 30:
                continue
            elif int(k) >= 20 and font.size < 25:
                continue
            img_list = gen_font_image5(font, v, disturb_list)
            if img_list is None:
                print (v, " : can not create. font=", font.path)
                continue

            font_name_size = font.path.replace('/', '_').replace('.', '_') + '_' + str(font.size)
            save_dir = os.path.join(sample_dir, k)
            my_mkdir(save_dir)
            for image in img_list:
                file_name = os.path.join(save_dir, '%s_%s_%d_%d.png' % (
                k, font_name_size, get_random(1, 10000), get_random(1, 10000)))
                cv2.imwrite(file_name, image)


def create_char_sample_train4():
    #sample_dir = 'chinese'
    sample_dir = '/Users/taoming/data/chineseOCR'
    char_file = 'hanzi_dict_reverse.txt'
    hanzi_dict = load_char_table(char_file)

    #font_dir = 'font'
    font_dir = '/Users/taoming/data/font'
    # files_name = ['方正书宋_GBK.TTF', '方正姚体简体.ttf', '方正黑体_GBK.TTF',
    #               '方正宋黑_GBK.ttf', '方正楷体简体.ttf', '方正水柱_GBK.ttf',
    #               '方正粗圆简体.ttf', '方正粗宋_GBK.ttf', '方正细倩_GBK.ttf',
    #               '方正细珊瑚_GBK.ttf', '方正细圆_GBK.TTF', '方正细等线_GBK.ttf',
    #               '方正行楷_GBK.ttf', '方正铁筋隶书简体.ttf', '方正魏碑_GBK.ttf']
    files_name = ['方正书宋_GBK.TTF']

    font_files = [os.path.join(font_dir, f) for f in files_name if True]

    # font_files = [ os.path.join(font_dir,f) for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir,f)) ]

    font_obj_list = []
    disturb_list = ['noise', 'filter', 'erosion', 'erosion2']
    # char_size=[30,40,50]
    # biaodian_size=[15,20,25,30]
    # char_size=[25, 30, 35, 40, 45, 50]
    char_size = [15, 20, 25, 30, 35, 45, 50]
    # char_size=[15, 20, 25, 30, 40, 50]
    # char_size=[15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 50]
    font_obj_list = []
    for size in char_size:
        for f in font_files:
            print ("f:" + f)
            print ("size:" + str(size))
            font = ImageFont.truetype(f, size)
            font_obj_list.append(font)

    for k, v in hanzi_dict.items():
        print (k)
        for font in font_obj_list:
            # 标点符号 数字 字体大小选为 15 ... 30
            if int(k) < 20 and font.size > 30:
                continue
            elif int(k) >= 20 and font.size < 25:
                continue
            img_list = gen_font_image5(font, v, disturb_list)
            if img_list is None:
                print (v, " : can not create. font=", font.path)
                continue

            font_name_size = font.path.replace('/', '_').replace('.', '_') + '_' + str(font.size)
            save_dir = os.path.join(sample_dir, k)
            my_mkdir(save_dir)
            for image in img_list:
                file_name = os.path.join(save_dir, '%s_%s_%d_%d.png' % (
                k, font_name_size, get_random(1, 10000), get_random(1, 10000)))
                cv2.imwrite(file_name, image)


def test_every_font():
    sample_dir = 'train/samples5'
    char_file = 'train/hanzi_table.txt'
    hanzi_dict = load_char_table(char_file)

    font_dir = 'train/font'
    font_files = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f))]
    font_obj_list = []

    disturb_list = ['noise', 'erosion', 'filter']
    char_size = [15, 50]
    # char_size=[15, 20, 25, 30, 35, 40, 45, 50]
    # char_size=[15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 50]
    font_obj_list = []
    for size in char_size:
        for f in font_files:
            font = ImageFont.truetype(f, size)
            font_obj_list.append(font)

    for k, v in hanzi_dict.items():
        for font in font_obj_list:
            for disturb in disturb_list:
                # font=font_obj_list[get_random(2,len(font_obj_list)-1)][get_random(0,len(font_files)-1)]
                image = gen_font_image3(font, v, disturb)
                if image is None:
                    print (v, " : can not create. font=", font.path)
                    continue

                font_name_size = font.path.replace('/', '_').replace('.', '_') + '_' + str(font.size) + '_' + disturb
                # save_dir=os.path.join(sample_dir, save_dir)
                save_dir = os.path.join(sample_dir, font_name_size)
                my_mkdir(save_dir)
                file_name = os.path.join(save_dir, '%s_%d_%d.png' % (k, get_random(1, 10000), get_random(1, 10000)))
                cv2.imwrite(file_name, image)


if __name__ == '__main__':
    # test_image_font()

    # bmp_to_png()
    # test_erosion()
    # test_erosion2()
    # create_char_sample_test()
    # create_char_sample_train()
    # create_char_sample_train2()
    # create_char_sample_train3()
    create_char_sample_train4()
    # test_every_font()
    # create_char_sample_test_mini()


