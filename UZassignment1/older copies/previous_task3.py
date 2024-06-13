
from UZ_utils import *

import numpy as np
import cv2
from matplotlib import pyplot as plt

def test_opps():
    I = imread(".\\images\\mask.png")
    # imshow(I)
    # height, width, channels = I.shape
    # print(I.shape)
    # print(I.dtype)

    I_gray = np.sum(I, axis=2) / 3
    # imshow(I_gray)
    plt.imshow(I_gray, cmap='gray')
    plt.show()

    I_gray = dilation(I_gray)
    plt.imshow(I_gray, cmap='gray')
    plt.show()

    I_gray = erosion(I_gray)
    plt.imshow(I_gray, cmap='gray')
    plt.show()

    I_gray = dilation(I_gray)
    plt.imshow(I_gray, cmap='gray')
    plt.show()

    I_gray = erosion(I_gray)
    plt.imshow(I_gray, cmap='gray')
    plt.show()

def myhist_dynamic(I_gray, num_of_bins):
    # boundaries = np.arange(0, num_of_bins, 1/num_of_bins)
    H = np.zeros(num_of_bins)
    pixel_vals = np.reshape(I_gray, -1)

    max = pixel_vals.max()
    min = pixel_vals.min()
    value_span = max - min

    divider = value_span/num_of_bins

    pixel_vals = pixel_vals-min
    pixel_vals = pixel_vals / divider
    pixel_vals = (np.floor(pixel_vals)).astype(int)
    unique_vals, counts = np.unique(pixel_vals, return_counts=True)

    for i in range(len(unique_vals)):
        if unique_vals[i] == num_of_bins:
            H[num_of_bins-1] += counts[i]
        else:
            H[unique_vals[i]] += counts[i]

    H = H / np.sum(H)
    return (H, min, divider)
    

def otsu_get_bin(my_histogram):
    # Returns the first bin on the right side of the divide.
    # Returns in the range 1 to (num_of_bins).
    # If the value is (num_of_bins) that means all bars are on the left.
    
    num_of_bins = len(my_histogram)
    max_var_between = -1
    best_T = -1
    for T in range(1, num_of_bins+1):
        var_between = T * (num_of_bins-T) * (np.mean(my_histogram[0:T]) - np.mean(my_histogram[T:]))**2
        if var_between > max_var_between:
            max_var_between = var_between
            best_T = T
    
    return best_T

def otsu_treshold(I_gray, num_of_bins):
    H, min, step = myhist_dynamic(I_gray, num_of_bins)
    bin = otsu_get_bin(H)
    
    # min + 1 * step means the first bin is on the left, and all the others on the right.
    # This is correct.
    # bin cannot be 0 because of how otsu_get_bin is built.
    # min + num_of_bins * step = max    (correct since    step = (max - min)/num_of_bins)
    treshold = min + bin * step
    return treshold

def treshold_mask(I_gray, treshold):
    I_mask = np.copy(I_gray)
    I_mask[I_gray < treshold] = 0
    I_mask[I_gray >= treshold] = 1

    return I_mask












def erosion(I_gray, n=5):
    SE = np.ones((n,n)) # create a square structuring element
    I_eroded = cv2.erode(I_gray, SE)
    return I_eroded

def dilation(I_gray, n=5):
    SE = np.ones((n,n)) # create a square structuring element
    I_dilated = cv2.dilate(I_gray, SE)
    return I_dilated

def opening(I_gray, n=5):
    I_opened = dilation(erosion(I_gray, n), n)
    return I_opened

def closing(I_gray, n=5):
    I_closed = erosion(dilation(I_gray, n), n)
    return I_closed




def main():

    
    
    
    
    # test_opps()





    I = imread(".\\images\\bird.jpg")
    # imshow(I)
    # height, width, channels = I.shape
    # print(I.shape)
    # print(I.dtype)

    I_gray = np.sum(I, axis=2) / 3
    # imshow(I_gray)
    plt.imshow(I_gray, cmap='gray')
    plt.show()

    I_bird_mask = treshold_mask(I_gray, otsu_treshold(I_gray, 1000))
    # I_bird_mask = treshold_mask(I_gray, 0.3)


    plt.imshow(I_bird_mask, cmap='gray')
    plt.show()

    I_bird_mask = opening(I_bird_mask, 3)
    
    plt.imshow(I_bird_mask, cmap='gray')
    plt.show()

    I_bird_mask = closing(I_bird_mask, 3)

    plt.imshow(I_bird_mask, cmap='gray')
    plt.show()












main()