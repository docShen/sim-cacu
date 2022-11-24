# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image
from numpy import average, dot, linalg
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def make_regalur_image(img, size=(64, 64)):
    gray_image = img.resize(size).convert('RGB')
    return gray_image

def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    hist = sum(1 - (0 if l == r else float(abs(l-r))/max(l,r))for l, r in zip(lh, rh))/len(lh)
    return hist


def calc_similar(li, ri):
    calc_sim = hist_similar(li.histogram(), ri.histogram())
    return calc_sim


def get_thum(image, greyscale=False):
    if greyscale:
        image = image.convert('L')
    return image

def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    res = dot(a / a_norm, b / b_norm)
    return res

def compute_similarity(img1_path, img2_path , lam1, lam2,lam3):

    image1 = Image.open(img1_path)
    image2 = Image.open(img2_path)

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # cos sim
    cosin = np.round(image_similarity_vectors_via_numpy(image1, image2), 4)

    # ssim
    ssim = np.round(compare_ssim(img1, img2, multichannel=True), 4)

    # hist sim
    hist_sim = np.round(calc_similar(make_regalur_image(image1), make_regalur_image(image2)), 4)

    # mse sim 注意：这个是图片的平均误差，不是百分数 ,越小表示越相似
    mse = np.round(compare_mse(img1, img2), 4)

    # psnr sim ： 注意 PSNR越大 表示图像越相似 ， 如果出现inf表示一模一样
    psnr = np.round(compare_psnr(img1, img2))

    # lam 1 ,2 ,2 求和为 1 ,表示不同weight


    assert lam1 + lam2 + lam3 == 1, 'weight error'

    result_sim = (lam1 * cosin + lam2 * ssim + lam3 * hist_sim)


    '''
    out put
    '''
    print('cos sim', cosin * 100, '%\n')
    print('structure sim ', ssim * 100, '%\n')
    print("hist sim", hist_sim * 100, "%\n")
    print("mse sim", mse,'->>>>>>>>>>>>越小表示越相似\n')
    print("psnr psnr", psnr,'->>>>>>>>>>>>越大表示越相似\n')

    return result_sim





if __name__ == '__main__':

    img1_path = '/home/coolshen/Desktop/code/mycode/img_similarity/mid.jpg'
    img2_path = '/home/coolshen/Desktop/code/mycode/img_similarity/mid.jpg'

    similarity = compute_similarity(img1_path,img2_path,lam1=0.5,lam2=0.2,lam3=0.3)


    print('weighted sim（加权相似度） ->>>> ', similarity * 100 ,'%')
