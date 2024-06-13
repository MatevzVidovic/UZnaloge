

from UZ_utils import *

import numpy as np
import cv2
from matplotlib import pyplot as plt

def cut_rectangle_monochannel(I, x: tuple, y: tuple, channel: int):
    return_array = I[x[0]:x[1], y[0]:y[1], channel]
    return return_array

def invert_subimage_float_inplace(I, x: tuple, y: tuple):
    # All channels there are get inverted.
    I[x[0]:x[1], y[0]:y[1], :] = 1 - I[x[0]:x[1], y[0]:y[1], :]
    return

def treshold_mask(I_gray, treshold):
    I_mask = np.copy(I_gray)
    I_mask[I_gray < treshold] = 0
    I_mask[I_gray >= treshold] = 1

    return I_mask

def myhist(I_gray, num_of_bins):
    # boundaries = np.arange(0, num_of_bins, 1/num_of_bins)
    H = np.zeros((2, num_of_bins))
    pixel_vals = np.reshape(I_gray, -1)

    H[0] = np.array(range(0, num_of_bins))

    divider = 1/num_of_bins
    pixel_vals = pixel_vals / divider
    pixel_vals = (np.floor(pixel_vals)).astype(int)
    unique_vals, counts = np.unique(pixel_vals, return_counts=True)
    for i in range(len(unique_vals)):
        if unique_vals[i] == num_of_bins:
            H[1][num_of_bins-1] += counts[i]
        else:
            H[1][unique_vals[i]] += counts[i]

    H[1] = H[1] / np.sum(H[1])
    return H

def myhist_dynamic(I_gray, num_of_bins):
    # boundaries = np.arange(0, num_of_bins, 1/num_of_bins)
    hist_counts = np.zeros(num_of_bins)
    pixel_vals = np.reshape(I_gray, -1)

    # This will have values of the groups of pixels in the 0-th row
    # And will have counts of those groups in the other.
    H = np.zeros((2, num_of_bins))
    
    max = pixel_vals.max()
    min = pixel_vals.min()
    value_span = max - min

    divider = value_span/num_of_bins

    pixel_vals = pixel_vals-min
    pixel_vals = pixel_vals / divider
    # We've basically converted each pixel value to k, where pixel value = k * divider + something less than a divider.
    # And k is also the index of the bin this pixel belonds to.
    

    print(min)
    print(divider*num_of_bins)
    print(divider/2)
    # we need to add the divider/2 to get the mean of the column correctly, not the left side (floored) of the column
    bar_values_of_pixels = [min + divider * i + divider/2 for i in range(0, num_of_bins)]
    H[0][:] = np.array(bar_values_of_pixels)
    print(H[0])

    pixel_vals = (np.floor(pixel_vals)).astype(int)
    unique_vals, counts = np.unique(pixel_vals, return_counts=True)

    for i in range(len(unique_vals)):

        # just for the chance that it's exactly equal to the maximum
        if unique_vals[i] == num_of_bins:
            hist_counts[num_of_bins-1] += counts[i]
            # H[0][num_of_bins-1] = min + divider * num_of_bins-1 + divider/2   
        else:
            hist_counts[unique_vals[i]] += counts[i]
            # we need to add the divider/2 to get the mean of the column correctly, not the left side (floored) of the column
            # H[0][unique_vals[i]] = min + divider * unique_vals[i] + divider/2   


    hist_counts = hist_counts / np.sum(hist_counts)
    H[1][:] = hist_counts[:]

    # print(H.shape)
    # print(H)

    return H

def otsu_get_bin(H):
    # Returns the index of the first bin on the right side of the divide.
    # Returns in the range 1 to (num_of_bins).
    # If the value is (num_of_bins) that means all bars are on the left.

    # https://en.wikipedia.org/wiki/Otsu%27s_method
    
    # https://www.youtube.com/watch?v=jUUkMaNuHP8



    my_histogram = H[1]

    num_of_bins = H.shape[1]
    max_var_between = -1
    best_T = -1
    for T in range(1, num_of_bins+1):

        # These are the number of pixels in the background / all pixels and likewise for foreground. 
        background_percentage = np.sum(my_histogram[0:T])
        foreground_percentage = np.sum(my_histogram[T:])

        background_weighted_values = H[0][0:T] * H[1][0:T]
        foreground_weighted_values = H[0][T:] * H[1][T:]

        # This is correct according to the YT formula.
        # This checks out, because in the numerator in each separate addition we already have the numbers of pixels divided by the numerus.
        # So when we divide by the num_of_pixels/numerus the numerus goes as a multiplyer to he numerator, so it cancels out.
        background_mean = np.sum(background_weighted_values)/background_percentage
        foreground_mean = np.sum(foreground_weighted_values)/foreground_percentage

        var_between = background_percentage * foreground_percentage - (background_mean - foreground_mean)**2
        if var_between > max_var_between:
            max_var_between = var_between
            best_T = T
    
    return best_T

def otsu_treshold(I_gray, num_of_bins):
    H = myhist_dynamic(I_gray, num_of_bins)
    bin = otsu_get_bin(H)
    
    #myhist_dynamic used to return (H, step, min) so you could calculate this treshold.
    # At that time H was only a vector of the values of the bars. It didn't have the current H[0] yet.

    # min + 1 * step means the first bin is on the left, and all the others on the right.
    # This is correct.
    # bin cannot be 0 because of how otsu_get_bin is built.
    # min + num_of_bins * step = max    (correct since    step = (max - min)/num_of_bins)
    # treshold = min + bin * step

    treshold = H[0][bin]

    return treshold


# Testne funkcije:

def test_dynamic_myhist():
    I = np.arange(0.4, 0.7, 0.001)
    num_of_bins = 100

    plt.clf()

    H = myhist(I, num_of_bins)
    H = H / np.sum(H)
    plt.subplot(2, 1, 1)
    plt.bar(range(len(H)), H[1], align='edge')

    H = myhist_dynamic(I, num_of_bins)
    plt.subplot(2, 1, 2)
    plt.bar(H[0], H[1], align='center', width=(1/(1.4*H.shape[1])))

    plt.show()

def plot_both_types_of_histograms(I_gray, num_of_bins = 100):
    plt.clf()
    H = myhist(I_gray, num_of_bins)
    H = H / np.sum(H)
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.bar(range(len(H)), H, align='edge')
    # plt.show()

    # print(H)
    H = myhist_dynamic(I_gray, num_of_bins)
    plt.subplot(2, 1, 2)
    # print(H)
    # plt.bar(range(len(H)), H[1], align='edge')
    plt.bar(H[0], H[1], align='center', width=(1/(1.4*H.shape[1])))
    plt.show()

def otsu_test():
    
    img_names = ["umbrellas.jpg", "bird.jpg", "candy.jpg", "eagle.jpg", "mask.png"]
    for name in img_names:
        path = ".\\images\\" + name
        I = imread(path)
        I_gray = np.sum(I, axis=2) / 3
        plt.subplot(2, 1, 1)
        plt.imshow(I_gray, cmap='gray')
        
        I_mask = treshold_mask(I_gray, otsu_treshold(I_gray, 1000))
        plt.subplot(2, 1, 2)
        plt.imshow(I_mask, cmap='gray')

        plt.show()

def otsu_test_moje_slike():
    
    img_names = ["temna.jpg", "srednja.jpg", "svetla.jpg", "20231023_181600.jpg"]
    for name in img_names:
        path = ".\\moje_slike\\" + name
        I = imread(path)
        I_gray = np.sum(I, axis=2) / 3
        plt.subplot(2, 1, 1)
        plt.imshow(I_gray, cmap='gray')
        
        I_mask = treshold_mask(I_gray, otsu_treshold(I_gray, 1000))
        plt.subplot(2, 1, 2)
        plt.imshow(I_mask, cmap='gray')

        plt.show()

def made_for_understanding_mydynamic_hist(I_gray):


    num_of_bins = 100
    H = myhist(I_gray, num_of_bins)
    H = H / np.sum(H)
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.bar(H.shape[1], H[1], align='edge')
    # plt.show()

    # print(H)

    H = myhist_dynamic(I_gray, num_of_bins)
    plt.subplot(2, 1, 2)
    plt.bar(range(H.shape[1]), H[1], align='edge')
    plt.bar(H[0], H[1], align='center')

    # wtf TO PA POL KR DELA A KAJ
    # AJAAAAAAAAAAAAAAAAAAAAA
    # MATPLOTLIB MA PRIVZETO ŠIRINO STOLPCEV O MOJ BOOOOG
    # IN TO JE TO KAR VIDIMO
    H[0] = H[0]*100
    plt.bar(H[0], H[1], align='center')
    # print(H)
    # plt.show()




    num_of_bins = 100
    # H = myhist(I_gray, num_of_bins)
    # H = H / np.sum(H)
    plt.clf()
    # plt.bar(range(len(H)), H, align='edge')
    # # plt.show()

    # # print(H)

    H = myhist_dynamic(I_gray, num_of_bins)
    plt.subplot(2, 1, 1)
    plt.bar(range(H.shape[1]), H[1], align='edge')
    # plt.bar(H[0], H[1], align='center')

    plt.subplot(2, 1, 2)
    plt.bar(H[0], H[1], align='center', width=(1/(1.4*H.shape[1])))
    # print(H)
    # plt.show()







def main():
    I = imread(".\\images\\umbrellas.jpg")
    # imshow(I)
    # height, width, channels = I.shape
    # print(I.shape)
    # print(I.dtype)

    I_gray = np.sum(I, axis=2) / 3
    # imshow(I_gray)
    plt.imshow(I_gray, cmap='gray')
    # plt.show()



    # print(I)
    plt.clf()
    plt.subplot(2, 2, 1)
    # print(I_red)
    I_red = cut_rectangle_monochannel(I, (130, 260), (240, 450), 0)
    # imshow(I_red)
    plt.imshow(I_red)
    plt.subplot(2, 2, 2)
    plt.imshow(I_red, cmap='gray')

    I_green = cut_rectangle_monochannel(I, (130, 260), (240, 450), 1)
    I_blue = cut_rectangle_monochannel(I, (130, 260), (240, 450), 2)
    plt.subplot(2, 2, 3)
    plt.imshow(I_green)
    plt.subplot(2, 2, 4)
    plt.imshow(I_blue)
    # plt.show()

    invert_subimage_float_inplace(I, (130, 260), (240, 450))
    plt.clf()
    plt.imshow(I)
    # plt.show()


    I_gray_reduced = I_gray * 0.3
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.imshow(I_gray, cmap='gray')
    plt.subplot(2, 1, 2)
    plt.imshow(I_gray_reduced, cmap='gray')
    # plt.show()

    plt.clf()
    plt.subplot(2, 1, 1)
    plt.imshow(I_gray, vmin=0, vmax=1, cmap='gray')
    plt.subplot(2, 1, 2)
    plt.imshow(I_gray_reduced, vmin=0, vmax=1, cmap='gray')
    # plt.show()









    I = imread(".\\images\\bird.jpg")
    I_gray = np.sum(I, axis=2) / 3
    treshold = 0.3
    I_mask = treshold_mask(I_gray, treshold)

    plt.clf()
    plt.subplot(2, 1, 1)
    plt.imshow(I)
    plt.subplot(2, 1, 2)
    plt.imshow(I_mask, cmap='gray')
    # plt.show()






    made_for_understanding_mydynamic_hist(I_gray)

    




    # test_dynamic_myhist()



    I1 = np.sum(imread(".\\moje_slike\\temna.jpg"), axis=2) / 3
    I2 = np.sum(imread(".\\moje_slike\\srednja.jpg"), axis=2) / 3
    I3 = np.sum(imread(".\\moje_slike\\svetla.jpg"), axis=2) / 3
    
    # plot_both_types_of_histograms(I1)
    # plot_both_types_of_histograms(I2)
    # plot_both_types_of_histograms(I3)


    T = otsu_get_bin(H)
    # print(T)


    otsu_test()
    # otsu_test_moje_slike()



























main()