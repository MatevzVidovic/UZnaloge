

from UZ_utils import *

import numpy as np
import cv2
from matplotlib import pyplot as plt


def treshold_mask(I_gray, treshold):
    I_mask = np.copy(I_gray)
    I_mask[I_gray < treshold] = 0
    I_mask[I_gray >= treshold] = 1

    return I_mask

def myhist(I_gray, num_of_bins):
    
    # H[0] will hold the x value of the bar.
    # H[1] will hold the normalized counts for each bar (their heights).
    H = np.zeros((2, num_of_bins))
    
    pixel_vals = np.reshape(I_gray, -1)
    min = 0
    max = 1
    value_span = max - min

    divider = 1/num_of_bins
    pixel_vals = pixel_vals / divider
    pixel_vals = (np.floor(pixel_vals)).astype(int)
    # We've converted each pixel value to an integer k, where:
    # pixel value = k * divider + something less than a divider.
    # And k is also the index of the bin this pixel belonds to.


    # we need to add the divider/2 to get the mean of the column correctly,
    # not the left side (floored) of the column
    bar_values_of_pixels = [min + divider * i + divider/2 for i in range(0, num_of_bins)]
    H[0][:] = np.array(bar_values_of_pixels)


    unique_vals, counts = np.unique(pixel_vals, return_counts=True)
    for i in range(len(unique_vals)):
        if unique_vals[i] == num_of_bins:
        # just for the chance that it's exactly equal to the maximum
            H[1][num_of_bins-1] += counts[i]
        else:
            H[1][unique_vals[i]] += counts[i]

    H[1] = H[1] / np.sum(H[1])
    return H

def myhist_dynamic(I_gray, num_of_bins):

    # H[0] will hold the x value of the bar.
    # H[1] will hold the normalized counts for each bar (their heights).
    H = np.zeros((2, num_of_bins))
    


    pixel_vals = np.reshape(I_gray, -1)
    max = pixel_vals.max()
    min = pixel_vals.min()
    value_span = max - min

    divider = value_span/num_of_bins
    pixel_vals = pixel_vals-min
    pixel_vals = pixel_vals / divider
    pixel_vals = (np.floor(pixel_vals)).astype(int)
    # We've converted each pixel value to an integer k, where:
    # pixel value = k * divider + something less than a divider.
    # And k is also the index of the bin this pixel belonds to.
    

    # we need to add the divider/2 to get the mean of the column correctly,
    # not the left side (floored) of the column
    bar_values_of_pixels = [min + divider * i + divider/2 for i in range(0, num_of_bins)]
    H[0][:] = np.array(bar_values_of_pixels)
    

    unique_vals, counts = np.unique(pixel_vals, return_counts=True)
    for i in range(len(unique_vals)):
    # just for the chance that it's exactly equal to the maximum
        if unique_vals[i] == num_of_bins:
            H[1][num_of_bins-1] += counts[i]
        else:
            H[1][unique_vals[i]] += counts[i]

    # normalization of the counts
    H[1] = H[1] / np.sum(H[1])

    return H

def plot_histogram(histogram):
    plt.clf()
    
    H = histogram
    plt.subplot(1, 1, 1)
    plt.title("Excercise 2, task (b)")
    plt.bar(H[0], H[1], align='center', width=(1/(1.4*H.shape[1])))
    # print(H)
    plt.show()

def otsu_get_bin(H):
    # Returns the index of the first bin on the right side of the divide.
    # Returns in the range 1 to (num_of_bins-1).
    # We disallowed the value (num_of_bins), which would mean all bars are on the left.
    # We decided to not count this as a valid treshold, because it doesn't actually separate anything.
    # It also seems that mathematically taking a single bar from the right would for sure make the varience smaller.
    # On top of that it causes warnings and division by zero in the code.

    # https://en.wikipedia.org/wiki/Otsu%27s_method
    
    # https://www.youtube.com/watch?v=jUUkMaNuHP8



    num_of_bins = H.shape[1]
    max_var_between = -1
    best_T = -1
    for T in range(1, num_of_bins):

        # These are the number of pixels in the background / all pixels and likewise for foreground. 
        background_percentage = np.sum(H[1][0:T])
        foreground_percentage = np.sum(H[1][T:])

        background_weighted_values = H[0][0:T] * H[1][0:T]
        foreground_weighted_values = H[0][T:] * H[1][T:]

        # This is correct according to the YT formula.
        # This checks out, because in the numerator in each separate addition we already have the numbers of pixels divided by the numerus.
        # So when we divide by the num_of_pixels/numerus the numerus goes as a multiplyer to he numerator, so it cancels out.
        background_mean = np.sum(background_weighted_values)/background_percentage
        foreground_mean = np.sum(foreground_weighted_values)/foreground_percentage



        var_between = background_percentage * foreground_percentage * (background_mean - foreground_mean)**2
        # if (T%50 == 0):
        #     print("\n")
        #     print("T = " + str(T))
        #     print("var_between = " + str(var_between))
        
        if var_between > max_var_between:
            max_var_between = var_between
            best_T = T
    
            # informative_string = """background_percentage
            # foreground_percentage
            # background_weighted_values.sum()
            # background_weighted_values.sum()
            # background_mean
            # foreground_mean
            # var_between
            # T"""

            # print(informative_string)
            # print(background_percentage)
            # print(foreground_percentage)
            # print(background_weighted_values.sum())
            # print(background_weighted_values.sum())
            # print(background_mean)
            # print(foreground_mean)
            # print(var_between)
            # print(T)


    return best_T

def otsu_treshold(I_gray, num_of_bins):
    H = myhist_dynamic(I_gray, num_of_bins)
    bin = otsu_get_bin(H)
    # bin = otsu_get_bin2(H, I_gray.size)
    
    # this dinds the value that separates the bars correctly.
    # It averages the centre values of the bars between which the treshold is.
    # We can see this as: (left_upper_bar_val+divider/2 + left_upper_bar_val-divider/2)/2
    treshold = (H[0][bin] + H[0][bin-1])/2

    return treshold





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



def invert_mask(I_mask):
    new_mask = 1-I_mask[:,:]
    return new_mask

def immask(I_three_channel, I_mask):
    
    new_shape = (I_mask.shape[0], I_mask.shape[1], 3)
    I_3chan_mask = np.zeros(new_shape)
    
    for i in range(3):
        I_3chan_mask[:,:,i] = I_mask[:,:]
    # print(I_3chan_mask)

    I_masked_three_channel = I_three_channel * I_3chan_mask
    return I_masked_three_channel



def show(I):
    plt.imshow(I, cmap='gray')
    plt.show()



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






# Excercise 3, task (a)
if False:

    I_maskJpeg = imread(".\\images\\mask.png")
    plt.subplot(2,2,1)
    plt.title("Basic image:")
    plt.imshow(I_maskJpeg, cmap='gray')

    I_maskJpeg_gray = np.sum(I_maskJpeg, axis=2) / 3
    plt.subplot(2,2,2)
    plt.title("Gray image:")
    plt.imshow(I_maskJpeg_gray, cmap='gray')

    I_maskJpeg_dilated = dilation(I_maskJpeg_gray)
    plt.subplot(2,2,3)
    plt.title("Dilated image:")
    plt.imshow(I_maskJpeg_dilated, cmap='gray')

    I_maskJpeg_eroded = erosion(I_maskJpeg_gray)
    plt.subplot(2,2,4)
    plt.title("Eroded image:")
    plt.imshow(I_maskJpeg_eroded, cmap='gray')
    plt.show()






# Exercise 3, task (b)
if False:
    I_bird = imread(".\\images\\bird.jpg")
    # imshow(I)
    # height, width, channels = I.shape
    # print(I.shape)
    # print(I.dtype)

    I_bird_gray = np.sum(I_bird, axis=2) / 3
    # imshow(I_gray)
    plt.imshow(I_bird_gray, cmap='gray')
    plt.show()

    I_bird_mask = treshold_mask(I_bird_gray, otsu_treshold(I_bird_gray, 256))
    # I_bird_mask = treshold_mask(I_gray, 0.3)
    plt.imshow(I_bird_mask, cmap='gray')
    plt.show()

    for i in range(3):
        I_bird_mask = closing(I_bird_mask, 7)
        plt.imshow(I_bird_mask, cmap='gray')
        plt.show()

        I_bird_mask = opening(I_bird_mask, 7)
        plt.imshow(I_bird_mask, cmap='gray')
        plt.show()




# Excercise 3, task (c) and (d)
if False:

    I_eagle = imread(".\\images\\eagle.jpg")
    I_eagle_gray = np.sum(I_eagle, axis=2) / 3
    I_eagle_mask = treshold_mask(I_eagle_gray, otsu_treshold(I_eagle_gray, 256))
    
    # I_eagle_mask = invert_mask(I_eagle_mask)
    I_eagle_masked = immask(I_eagle, I_eagle_mask)

    # imshow(I_eagle_masked)
    plt.imshow(I_eagle_masked)
    plt.show()



    I_eagle = imread(".\\moje_slike\\svetla.jpg")
    I_eagle_gray = np.sum(I_eagle, axis=2) / 3
    I_eagle_mask = treshold_mask(I_eagle_gray, otsu_treshold(I_eagle_gray, 256))
    
    # I_eagle_mask = invert_mask(I_eagle_mask)
    I_eagle_masked = immask(I_eagle, I_eagle_mask)

    # imshow(I_eagle_masked)
    plt.imshow(I_eagle_masked)
    plt.show()







# Excercise 3, task (e)

if True:

    I_coins = imread(".\\images\\coins.jpg")
    I_coins_gray = np.sum(I_coins, axis=2) / 3
    I_coins_mask = treshold_mask(I_coins_gray, otsu_treshold(I_coins_gray, 256))
    I_opened = opening(I_coins_mask, 7)
    I_opened = opening(I_opened, 9)

    I_opened = invert_mask(I_opened)

    I_opened = I_opened.astype('uint8')
    returned_data = cv2.connectedComponentsWithStats(I_opened)
    num_of_components = returned_data[0]
    labels = returned_data[1]
    stats = returned_data[2]
    centroids = returned_data[3]

    for i in range (1, num_of_components):
        if(stats[(i, cv2.CC_STAT_AREA)] > 700):
            ix_left = stats[(i, cv2.CC_STAT_LEFT)]
            ix_right = stats[(i, cv2.CC_STAT_LEFT)] + stats[(i, cv2.CC_STAT_WIDTH)]
            ix_top = stats[(i, cv2.CC_STAT_TOP)]
            ix_bottom = stats[(i, cv2.CC_STAT_TOP)] + stats[(i, cv2.CC_STAT_HEIGHT)]
            
            # tole pa ne dela for some reason:
            # I_opened[ix_top:ix_bottom][ix_left:ix_right] = 0
            I_opened[ix_top:ix_bottom, ix_left:ix_right] = 0

            """
            stats = CC_STAT_LEFT Python: cv.CC_STAT_LEFT
            The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
            CC_STAT_TOP Python: cv.CC_STAT_TOP
            CC_STAT_WIDTH 
            Python: cv.CC_STAT_WIDTH
            The horizontal size of the bounding box.
            CC_STAT_HEIGHT 
            Python: cv.CC_STAT_HEIGHT
            The vertical size of the bounding box.
            CC_STAT_AREA The total area (in pixels)
            """
            

    imshow(I_opened)

    I_small_coins = immask(I_coins, I_opened)
    imshow(I_small_coins)