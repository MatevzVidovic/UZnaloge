
import numpy as np
import a3_utils as a3u
import UZ_utils
import matplotlib.pyplot as plt
import cv2
import math
import os

def gaussdx(sigma: float):
    side_width = int(math.ceil(3*sigma))
    kernel_size = int(2 * side_width +1)
    kernel_x_values = np.array(range(-side_width, side_width+1))
    coef = -1 / ((sigma ** 3) * math.sqrt(2*math.pi))
    kernel =  coef * kernel_x_values * np.exp( -(kernel_x_values ** 2) / (2* (sigma**2)) )
    
    kernel_abs = np.abs(kernel)
    kernel /= np.sum(kernel_abs)
    return kernel, kernel_x_values

def gauss(sigma: float):
    side_width = int(math.ceil(3*sigma))
    kernel_size = int(2 * side_width +1)
    kernel_x_values = np.array(range(-side_width, side_width+1))
    coef = 1 / (sigma * math.sqrt(2*math.pi))
    kernel =  coef * np.exp( -(kernel_x_values ** 2) / (2* (sigma**2)) )
    kernel /= np.sum(kernel)
    return kernel, kernel_x_values


def gauss_kern_horizontal_faux2D(sigma: float):
    ker = gauss(sigma)[0].reshape(1, -1)
    return ker

def gaussdx_kern_horizontal_faux2D(sigma: float):
    ker = gaussdx(sigma)[0].reshape(1, -1)
    return ker


def image_partial_derivatives(image, sigma: float):
    x_gauss = gauss_kern_horizontal_faux2D(sigma)
    y_gauss = x_gauss.T
    x_gauss_der = gaussdx_kern_horizontal_faux2D(sigma)
    y_gauss_der = x_gauss_der.T

    x_partial = UZ_utils.convolve(image, y_gauss, x_gauss_der)
    y_partial = UZ_utils.convolve(image, x_gauss, y_gauss_der)
    return x_partial, y_partial

def image_second_partial_ders(image, sigma: float):
    x_partial, y_partial = image_partial_derivatives(image, sigma)
    
    x_second_partial, xy_second_partial = image_partial_derivatives(x_partial, sigma)
    _, y_second_partial = image_partial_derivatives(y_partial, sigma)

    return x_second_partial, xy_second_partial, y_second_partial

def gradient_magnitude(image, sigma):
    x_partial, y_partial = image_partial_derivatives(image, sigma)

    magnitudes = np.sqrt(x_partial **2 + y_partial **2)
    angles = np.arctan2(y_partial, x_partial)

    return magnitudes, angles


def L2_distance(hist1, hist2):
    hist_diff = hist1 - hist2
    hist_dist = math.sqrt(np.sum(hist_diff ** 2))
    return hist_dist

def chi_square_distance(hist1, hist2):
    hist_dist_numerator = (hist1 - hist2) ** 2
    hist_dist_denominator = hist1 + hist2 + 1e-10 * np.ones(hist1.shape)

    hist_dist = 1/2 * np.sum(hist_dist_numerator / hist_dist_denominator)
    return hist_dist

def intersection_distance(hist1, hist2):
    min_hist = np.minimum(hist1, hist2)
    hist_dist = 1 - np.sum(min_hist)
    return hist_dist

def hellinger_distance(hist1, hist2):
    hist_root_diff = hist1 ** (1/2) - hist2 ** (1/2)
    hist_dist = (1/2 * np.sum(hist_root_diff ** 2)) ** (1/2)
    return hist_dist


def myhist3(image_gray, num_of_bins=16):
    # image_gray = np.sum(image_NOT_gray/3, axis=2)
    # image_gray.astype('float64')
    # print(image_gray)
    sigma = 1

    
    theta_min = -np.pi
    theta_max = np.pi
    theta_step = (theta_max - theta_min) / 8

    gradients, angles = gradient_magnitude(image_gray, sigma)
    max_y, max_x = gradients.shape
    y_size_of_grid_cell = int(math.floor(max_y / 8))
    x_size_of_grid_cell = int(math.floor(max_x / 8))

    grid_base_ys = [y_size_of_grid_cell * i for i in range(8)]
    grid_base_ys.append(max_y)
    grid_base_xs = [x_size_of_grid_cell * i for i in range(8)]
    grid_base_xs.append(max_x)

    hist = np.zeros((num_of_bins, num_of_bins, num_of_bins))
    for y_base_ix in range(len(grid_base_ys)-1):
        for x_base_ix in range(len(grid_base_xs)-1):
            if(y_base_ix == 7 and x_base_ix == 7):
                print("lemme see")
            for y_ix in range(grid_base_ys[y_base_ix+1] - grid_base_ys[y_base_ix]):
                for x_ix in range(grid_base_xs[x_base_ix+1] - grid_base_xs[x_base_ix]):
                    angle = angles[y_base_ix+y_ix, x_base_ix+x_ix]
                    angle_quantized_ix = int(math.floor(((angle - theta_min) / theta_step)))
                    # in case the angle is exactly pi
                    if(angle_quantized_ix == 8):
                        angle_quantized_ix = 7
                    currGrad = gradients[y_base_ix+y_ix, x_base_ix+x_ix]
                    hist[y_base_ix, x_base_ix, angle_quantized_ix] += currGrad
                    print(hist[y_base_ix, x_base_ix, angle_quantized_ix])
    hist = hist / hist.sum()
    return hist




num_of_bins = 8

base_path = ".\\dataset"
dir_list = os.listdir(base_path)
# print(dir_list)

images = []
for name in dir_list:
    read_image = UZ_utils.imread_gray(base_path + "\\" + name)
    images.append(read_image)


hists_3D = []
for ix in range(len(images)):
    if(ix == 19):
        print(myhist3(images[ix]))
        print("lemme see")

hists_1D_C_ordering = []
for hist in hists_3D:
    hists_1D_C_ordering.append(hist.reshape(-1))


if False:


    list_of_lists = [dir_list, images, hists_3D, hists_1D_C_ordering]
    chosen_ix = 19

    chosen_img = images[chosen_ix]
    chosen_3D_histogram = hists_3D[chosen_ix]
    chosen_1D_histogram = hists_1D_C_ordering[chosen_ix]





    L2_distances_with_ixs = []
    chi_square_distances_with_ixs = []
    intersection_distances_with_ixs = []
    hellinger_distances_with_ixs = []

    for ix in range(len(hists_1D_C_ordering)):
        hist = hists_1D_C_ordering[ix]
        L2_distances_with_ixs.append((L2_distance(hist, chosen_1D_histogram), ix))
        chi_square_distances_with_ixs.append((chi_square_distance(hist, chosen_1D_histogram), ix))
        intersection_distances_with_ixs.append((intersection_distance(hist, chosen_1D_histogram), ix))
        hellinger_distances_with_ixs.append((hellinger_distance(hist, chosen_1D_histogram), ix))

    distances_lists = [L2_distances_with_ixs, chi_square_distances_with_ixs, intersection_distances_with_ixs, hellinger_distances_with_ixs]
    sorted_distances_lists = []
    for given_list in distances_lists:
        sorted_distances_lists.append(sorted(given_list)) #key=lambda pair: pair[0]

    # distance_short_names = ["L2", "chi", "inter", "hell"]
    for sorted_dist_list in sorted_distances_lists:
        plt.subplot(2, 6, 1)
        plt.title(dir_list[chosen_ix])
        plt.imshow(chosen_img)
        plt.subplot(2, 6, 7)
        plt.plot(chosen_1D_histogram)

        for i in range(5):
            distance, ix = sorted_dist_list[i]
            plt.subplot(2, 6, 2+i)
            plt.title(dir_list[ix])
            plt.imshow(images[ix])
            plt.subplot(2, 6, 8+i)
            plt.title("{:.2f}".format(distance))
            plt.plot(hists_1D_C_ordering[ix])
        
        plt.show()

    for i in range(len(distances_lists)):
        data = [datapoint[0] for datapoint in distances_lists[i]]
        data_sorted = [datapoint[0] for datapoint in sorted_distances_lists[i]]
        
        best_ixs = [datapoint[1] for datapoint in sorted_distances_lists[i][0:5]]
        best_datums = [datapoint[0] for datapoint in sorted_distances_lists[i][0:5]]
        
        plt.subplot(1, 2, 1)
        plt.plot(data)
        plt.scatter(best_ixs, best_datums, marker="o", facecolors="none", edgecolors="k")
        plt.subplot(1, 2, 2)
        plt.plot(data_sorted)
        plt.scatter(range(5), best_datums, marker="o", facecolors="none", edgecolors="k")
        plt.show()