import numpy as np
import a3_utils as a3u
import UZ_utils
import matplotlib.pyplot as plt
import cv2
import math
import os
import random


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




impulse = np.zeros((50, 50))
impulse[25, 25] = 1

sigma = 4

G = gauss(sigma)[0].reshape(1, -1)
D = gaussdx(sigma)[0].reshape(1, -1)


plt.subplot(2, 3, 1)
plt.title("Impulse")
plt.imshow(impulse, cmap="gray")

current_image = UZ_utils.convolve(impulse, G, G.T)
plt.subplot(2, 3, 2)
plt.title("G, G.T")
plt.imshow(current_image, cmap="gray")

current_image = UZ_utils.convolve(impulse, G, D.T)
plt.subplot(2, 3, 3)
plt.title("G, D.T")
plt.imshow(current_image, cmap="gray")

current_image = UZ_utils.convolve(impulse, D, G.T)
plt.subplot(2, 3, 4)
plt.title("D, G.T")
plt.imshow(current_image, cmap="gray")

current_image = UZ_utils.convolve(impulse, G.T, D)
plt.subplot(2, 3, 5)
plt.title("G.T, D")
plt.imshow(current_image, cmap="gray")

current_image = UZ_utils.convolve(impulse, D.T, G)
plt.subplot(2, 3, 6)
plt.title("D, G")
plt.imshow(current_image, cmap="gray")

plt.show()




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




original_image = UZ_utils.imread_gray(".\images\museum.jpg")
sigma = 1

plt.subplot(2, 4, 1)
plt.title("Original")
plt.imshow(original_image, cmap="gray")


I_x, I_y = image_partial_derivatives(original_image, sigma)
plt.subplot(2, 4, 2)
plt.title("I_x")
plt.imshow(I_x, cmap="gray")

plt.subplot(2, 4, 3)
plt.title("I_y")
plt.imshow(I_y, cmap="gray")

I_xx, I_xy, I_yy = image_second_partial_ders(original_image, sigma)
plt.subplot(2, 4, 4)
plt.title("I_xx")
plt.imshow(I_xx, cmap="gray")

plt.subplot(2, 4, 5)
plt.title("I_xy")
plt.imshow(I_xy, cmap="gray")

plt.subplot(2, 4, 6)
plt.title("I_yy")
plt.imshow(I_yy, cmap="gray")


magnitudes, angles = gradient_magnitude(original_image, sigma)
plt.subplot(2, 4, 7)
plt.title("magnitudes")
plt.imshow(magnitudes, cmap="gray")

plt.subplot(2, 4, 8)
plt.title("angles")
plt.imshow(angles, cmap="gray")

plt.show()








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


def myhist3(image_gray, num_of_bins=8):
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
    grid_base_ys.append(max_x)

    hist = np.zeros((num_of_bins, num_of_bins, num_of_bins))
    for y_base_ix in range(len(grid_base_ys)-1):
        for x_base_ix in range(len(grid_base_xs)-1):
            for y_ix in range(grid_base_ys[y_base_ix+1] - grid_base_ys[y_base_ix]):
                for x_ix in range(grid_base_xs[x_base_ix+1] - grid_base_xs[x_base_ix]):
                    angle = angles[grid_base_ys[y_base_ix]+y_ix, grid_base_xs[x_base_ix]+x_ix]
                    angle_quantized_ix = int(math.floor(((angle - theta_min) / theta_step)))
                    # in case the angle is exactly pi
                    if(angle_quantized_ix == 8):
                        angle_quantized_ix = 7
                    current_gradient = gradients[grid_base_ys[y_base_ix]+y_ix, grid_base_xs[x_base_ix]+x_ix]
                    hist[y_base_ix, x_base_ix, angle_quantized_ix] += current_gradient
    
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
for img in images:
    hists_3D.append(myhist3(img))

hists_1D_C_ordering = []
for hist in hists_3D:
    hists_1D_C_ordering.append(hist.reshape(-1))





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



def findedges(image, sigma: float, theta: float):
    magnitudes, _ = gradient_magnitude(image, sigma)
    mag_mask = np.zeros(magnitudes.shape)
    mag_mask[magnitudes >= theta] = 1
    return mag_mask

museum = UZ_utils.imread_gray(".\\images\\museum.jpg")

for i in np.arange(0.1, 0.25, 0.1):
    edge_mask = findedges(museum, 1, i)
    plt.imshow(edge_mask, cmap="gray")
    plt.show()









def non_maxima_surpressed(theta_mask, magnitudes, angles):
    """
    Takes theta_mask, magnitudes, angles. Returns binary mask of true edge pixels.
    """

    padding_column = np.zeros((theta_mask.shape[0], 1))
    zero_padded_mask = np.concatenate((padding_column, theta_mask, padding_column), axis=1)
    padding_row = np.zeros((1, zero_padded_mask.shape[1]))
    zero_padded_mask = np.concatenate((padding_row, zero_padded_mask, padding_row), axis=0)
    # print("Should be (2, 2): (" + str(zero_padded_mask.shape[0] - theta_mask.shape[0]) + ", " + str(zero_padded_mask.shape[1] - theta_mask.shape[1]) + ")")
    # print(zero_padded_mask)
    zero_padded_magnitudes = np.concatenate((padding_column, magnitudes, padding_column), axis=1)
    zero_padded_magnitudes = np.concatenate((padding_row, zero_padded_magnitudes, padding_row), axis=0)

    zero_padded_angles = np.concatenate((padding_column, angles, padding_column), axis=1)
    zero_padded_angles = np.concatenate((padding_row, zero_padded_angles, padding_row), axis=0)

    
    
    final_mask = np.zeros(theta_mask.shape)

    # print("zero_padded_mask, zero_padded_magnitudes, zero_padded_angles, final_mask")
    # print(zero_padded_mask, zero_padded_magnitudes, zero_padded_angles, final_mask)


    
    
    # the y component comes first in numpy, so I made this dictionary incorrectly, because you wouldn't
    # generally write it like this in math
    chosen_neighbour2relative_ix_tuple = {
        0 : (0, 1),
        1 : (1, 1),
        2 : (1, 0),
        3 : (1, -1),
        4 : (0, -1),
        5 : (-1, -1),
        6 : (-1, 0),
        7 : (-1, 1),
    }

    #Wrong one:
    # chosen_neighbour2relative_ix_tuple = {
    #     0 : (1, 0),
    #     1 : (1, 1),
    #     2 : (0, 1),
    #     3 : (-1, 1),
    #     4 : (-1, 0),
    #     5 : (-1, -1),
    #     6 : (0, -1),
    #     7 : (1, -1),
    # }



    
    # y_ix comes first in numpy notation. I messed this up the first time, and had problems with how I constructed the dictionary.
    # Just watch out is all I'm trying to say.

    for y_ix in range(1, zero_padded_mask.shape[0]-1):
        for x_ix in range(1, zero_padded_mask.shape[1]-1):

            if(theta_mask[y_ix-1, x_ix-1] == 1):

                # the angle ranges from -pi to pi dependant on the x axis ray
                """
                We are going to do it like this:
                Add half of the angle span an individual neighbour takes (2*np.pi/8 /2) - 
                - this shifts the angle values and makes division possible.
                
                If it's still a negative angle, add 2.pi to it.
                (this order of doing it makes it easier. And even clearer.)

                Then simply do: math.floor(val / (2*np.pi / 8)). This gives you which of the neighbours,
                (going clockwise, starting with the right neighbour, starting indexing by 0)
                the angle is pointing to.
                """
                
                current_angle = zero_padded_angles[y_ix, x_ix]
                current_angle += ((2*np.pi / 8) / 2)
                if(current_angle < 0):
                    current_angle += 2*np.pi

                chosen_neighbour = int(math.floor(current_angle / (2*np.pi / 8)))
                relative_ix_tuple = chosen_neighbour2relative_ix_tuple[chosen_neighbour]
                # because diametrical neigbours have the sam L2 norm, we can just do this:
                diametrical_neighbour_relative_ix_tuple = (-relative_ix_tuple[0], -relative_ix_tuple[1])

                current_magnitude = zero_padded_magnitudes[y_ix, x_ix]
                neighbour_1_mag = zero_padded_magnitudes[y_ix + relative_ix_tuple[0], x_ix + relative_ix_tuple[1]]
                neighbour_2_mag = zero_padded_magnitudes[y_ix + diametrical_neighbour_relative_ix_tuple[0], x_ix + diametrical_neighbour_relative_ix_tuple[1]]

                if(current_magnitude >= neighbour_1_mag and current_magnitude >= neighbour_2_mag):
                    final_mask[y_ix-1, x_ix-1] = 1

                    # print("x_ix-1, y_ix-1, relative_ix_tuple, diametrical_neighbour_relative_ix_tuple")
                    # print(x_ix-1, y_ix-1, relative_ix_tuple, diametrical_neighbour_relative_ix_tuple)

                
    # print("final_mask")
    # print(final_mask)

    return final_mask










museum_img_gray = UZ_utils.imread_gray(".\images\museum.jpg")

sigma = 1
theta = 0.15

magnitudes, angles = gradient_magnitude(museum_img_gray, sigma)
museum_edges = findedges(museum_img_gray, sigma, theta)
# print("museum_edges, magnitudes, angles:")
# print(museum_edges, magnitudes, angles)
surpressed_edges = non_maxima_surpressed(museum_edges, magnitudes, angles)

# print("surpressed_edges")
# print(surpressed_edges)

# plt.subplot(1, 2, 1)
plt.imshow(museum_edges, cmap="gray")
plt.show()

# plt.subplot(1, 2, 2)
plt.imshow(surpressed_edges, cmap="gray")

plt.show()





def hysteresis(img_gray, sigma, theta_low, theta_high):

    magnitudes, angles = gradient_magnitude(img_gray, sigma)

    img_edges_low = findedges(img_gray, sigma, theta_low)
    img_edges_high = findedges(img_gray, sigma, theta_high)

    # plt.imshow(img_edges_low, cmap="gray")
    # plt.show()
    # plt.imshow(img_edges_high, cmap="gray")
    # plt.show()
    
    surpressed_edges_low = non_maxima_surpressed(img_edges_low, magnitudes, angles)
    surpressed_edges_high = non_maxima_surpressed(img_edges_high, magnitudes, angles)

    plt.imshow(surpressed_edges_low, cmap="gray")
    plt.show()
    plt.imshow(surpressed_edges_high, cmap="gray")
    plt.show()

    surpressed_edges_high = surpressed_edges_high.astype('uint8')
    num_of_comps_high, labels_high, _, _ = cv2.connectedComponentsWithStats(surpressed_edges_high, connectivity=8)


    surpressed_edges_low = surpressed_edges_low.astype('uint8')
    num_of_comps_low, labels_low, _, _ = cv2.connectedComponentsWithStats(surpressed_edges_low, connectivity=8)

    # plt.imshow(labels_low)
    # plt.show()
    # plt.imshow(labels_high)
    # plt.show()
    
    # pomoje se 0, ki je background, šteje med num of compoinents, in je torej spodnji range pravilen
    # all_nums_low = set(range(1, num_of_comps_low))
    # print(all_nums_low)

    accepted_nums_low = set()

    for y_ix in range(labels_low.shape[0]):
        for x_ix in range(labels_low.shape[1]):
            if labels_high[y_ix, x_ix] != 0 and labels_low[y_ix, x_ix] != 0:
                if not labels_low[y_ix, x_ix] in accepted_nums_low:
                    accepted_nums_low.add(labels_low[y_ix, x_ix])

    for y_ix in range(labels_low.shape[0]):
            for x_ix in range(labels_low.shape[1]):
                if labels_low[y_ix, x_ix] != 0:
                    if not labels_low[y_ix, x_ix] in accepted_nums_low:
                        surpressed_edges_low[y_ix, x_ix] = 0
    
    return surpressed_edges_low








museum_img_gray = UZ_utils.imread_gray(".\images\museum.jpg")
# plt.imshow(museum_img_gray, cmap="gray")
# plt.show()
    
# zgoraj theta = 0.15
hysterised = hysteresis(museum_img_gray, 1, 0.04, 0.16)
plt.imshow(hysterised, cmap="gray")
plt.show()





def hough_graph_of_point(x, y, max_y, max_x, num_of_bins):

    # max r for sure cannot be more than: max_y + max_x, and it cannot be less than -(max_y + max_x)
    # r_one_cell_diff = 2 * (max_y + max_x) / num_of_bins
    # max_r = max_y + max_x
    # min_r = - max_r

    """This might be causing problems. I should make the max r the length of the diagonal."""
    max_r = math.sqrt(max_y**2 + max_x**2)
    min_r = - max_r
    r_one_cell_diff = (max_r - min_r) / num_of_bins



    accumulator = np.zeros((num_of_bins, num_of_bins))
    
    max_theta = np.pi / 2
    min_theta = -max_theta
    theta_one_cell_diff = (max_theta - min_theta) / num_of_bins


    # theta = np.linspace(-np.pi, np.pi, num_of_bins)
    for theta_ix in range (num_of_bins):
        theta = theta_ix * theta_one_cell_diff + min_theta

        r = x * np.cos(theta) + y * np.sin(theta)
        r_translated = r + min_r
        r_ix = int(math.floor(r_translated / r_one_cell_diff))

        accumulator[r_ix, theta_ix] += 1
    
    return accumulator


point_1 = (10, 10)
point_2 = (30, 60)
point_3 = (50, 20)
point_4 = (80, 90)

max_coordinates = (100, 100)

plt.subplot(2, 2, 1)
plt.imshow(hough_graph_of_point(*point_1, *max_coordinates, 300))

plt.subplot(2, 2, 2)
plt.imshow(hough_graph_of_point(*point_2, *max_coordinates, 300))

plt.subplot(2, 2, 3)
plt.imshow(hough_graph_of_point(*point_3, *max_coordinates, 300))

plt.subplot(2, 2, 4)
plt.imshow(hough_graph_of_point(*point_4, *max_coordinates, 300))
plt.show()









def hough_find_lines(image, r_num_of_bins, theta_num_of_bins, treshold=0):
    # max r for sure cannot be more than: max_y + max_y, and it cannot be less than -(max_y + max_x)

    accumulator = np.zeros((r_num_of_bins, theta_num_of_bins))
    
    theta_one_cell_diff = np.pi / theta_num_of_bins
    min_theta = -np.pi / 2

    max_y, max_x = image.shape
    max_r = math.sqrt(max_y**2 + max_x**2)
    min_r = - max_r
    r_one_cell_diff = (max_r - min_r) / r_num_of_bins

    # theta = np.linspace(-np.pi, np.pi, num_of_bins)
    for y_ix in range(image.shape[0]):
        for x_ix in range(image.shape[1]):
            if(image[y_ix, x_ix] == 1):

                for theta_ix in range(theta_num_of_bins):
                    theta = theta_ix * theta_one_cell_diff + min_theta

                    r = x_ix * np.cos(theta) + y_ix * np.sin(theta)
                    r_translated = r + min_r
                    r_ix = int(math.floor(r_translated / r_one_cell_diff))

                    accumulator[r_ix, theta_ix] += 1
    
    return accumulator

synthetic_image = np.zeros((100, 100))
synthetic_image[10, 10] = 1
synthetic_image[10, 20] = 1
acc = hough_find_lines(synthetic_image, 300, 300, 0)
plt.imshow(acc)
plt.show()

# print(acc)

oneline_img = UZ_utils.imread_gray(".\\images\\oneline.png")
oneline_edges = findedges(oneline_img, 1, 0.2)
plt.imshow(oneline_edges)
plt.show()
acc = hough_find_lines(oneline_edges, 300, 300, 0)
plt.imshow(acc)
plt.show()


rect_img = UZ_utils.imread_gray(".\\images\\rectangle.png")
rect_edges = findedges(rect_img, 1, 0.2)
plt.imshow(rect_edges)
plt.show()
acc = hough_find_lines(rect_edges, 300, 300, 0)
plt.imshow(acc)
plt.show()








def nonmaxima_suppression_box(hough_2d_array):
    return_array = hough_2d_array.copy()
    for y in range(hough_2d_array.shape[0]):
        for x in range(hough_2d_array.shape[1]):
            left_ix = x-1 if x-1 >= 0 else 0
            right_ix = x+1 if x+1 <= (hough_2d_array.shape[1]-1) else (hough_2d_array.shape[1]-1)
            top_ix = y-1 if y-1 >= 0 else 0
            bottom_ix = y+1 if y+1 <= (hough_2d_array.shape[0]-1) else (hough_2d_array.shape[0]-1)

            # print(hough_2d_array)
            # print(left_ix, right_ix, top_ix, bottom_ix)
            neighbourhood = hough_2d_array[top_ix:bottom_ix+1, left_ix:right_ix+1]
            # print(neighbourhood)
            neighbourhood = neighbourhood.reshape(-1)
            # print(neighbourhood)
            neighbourhood = np.sort(neighbourhood)
            # print(neighbourhood)

            
            if neighbourhood[-1] != hough_2d_array[y, x]:
                return_array[y, x] = 0
    
    return return_array


oneline_img = UZ_utils.imread_gray(".\\images\\oneline.png")
oneline_edges = findedges(oneline_img, 1, 0.2)
plt.imshow(oneline_edges)
plt.show()
acc = hough_find_lines(oneline_edges, 300, 300, 0)
plt.imshow(acc)
plt.show()
acc_surpresed = nonmaxima_suppression_box(acc)
plt.imshow(acc_surpresed)
plt.show()  

rect_img = UZ_utils.imread_gray(".\\images\\rectangle.png")
rect_edges = findedges(rect_img, 1, 0.2)
plt.imshow(rect_edges)
plt.show()
acc = hough_find_lines(rect_edges, 300, 300, 0)
plt.imshow(acc)
plt.show()
acc_surpresed = nonmaxima_suppression_box(acc)
plt.imshow(acc_surpresed)
plt.show()  









def hough_tresholded(parameter_space, treshold):
    """takes parameter space with ixs: (r, theta).
    Returns pairs (r, theta) of pixesls with count above the treshold."""

    copied_parameter_space = parameter_space.copy()
    copied_parameter_space[parameter_space <= treshold] = 0

    return copied_parameter_space


def hough_treshold_lines(parameter_space, treshold):
    """takes parameter space with ixs: (r, theta).
    Returns pairs (r, theta) of pixesls with count above the treshold."""

    copied_parameter_space = parameter_space.copy()
    copied_parameter_space[parameter_space <= treshold] = 0

    # UZ_utils.imshow(copied_parameter_space)

    return_pairs = list()
    for r in range(copied_parameter_space.shape[0]):
        for theta in range(copied_parameter_space.shape[1]):
            if copied_parameter_space[r, theta] > 0:
                return_pairs.append((r, theta))
    
    return return_pairs





# !!!!! this is actually the oneline example

rect_img = UZ_utils.imread_gray(".\\images\\oneline.png")
rect_edges = findedges(rect_img, 1, 0.2)
plt.imshow(rect_edges)
plt.show()


theta_num_of_bins = 700
r_num_of_bins = theta_num_of_bins

acc = hough_find_lines(rect_edges, r_num_of_bins, theta_num_of_bins, 0)
plt.imshow(acc)
plt.show()
acc_surpresed = nonmaxima_suppression_box(acc)
plt.imshow(acc_surpresed)
plt.show()

rect_with_lines = rect_img.copy()
okay_lines = hough_treshold_lines(acc_surpresed, 100)
# print(okay_lines)
plt.imshow(rect_with_lines)




theta_one_cell_diff = np.pi / theta_num_of_bins
min_theta = -np.pi / 2

max_y, max_x = rect_with_lines.shape
max_r = math.sqrt(max_y**2 + max_x**2)
min_r = - max_r
r_one_cell_diff = (max_r - min_r) / r_num_of_bins

for i in okay_lines:
    # print(i[0] * r_one_cell_diff + min_r, i[1] * theta_one_cell_diff + min_theta)
    a3u.draw_line(i[0] * r_one_cell_diff + min_r, i[1] * theta_one_cell_diff + min_theta , *rect_with_lines.shape)
plt.show()









rect_img = UZ_utils.imread_gray(".\\images\\rectangle.png")
rect_edges = findedges(rect_img, 1, 0.2)
plt.imshow(rect_edges)
plt.show()



acc = hough_find_lines(rect_edges, r_num_of_bins, theta_num_of_bins, 0)
plt.imshow(acc)
plt.show()
acc_surpresed = nonmaxima_suppression_box(acc)
plt.imshow(acc_surpresed)
plt.show()

rect_with_lines = rect_img.copy()
okay_lines = hough_treshold_lines(acc_surpresed, 150)
# print(okay_lines)
plt.imshow(rect_with_lines)




theta_one_cell_diff = np.pi / theta_num_of_bins
min_theta = -np.pi / 2

max_y, max_x = rect_with_lines.shape
max_r = math.sqrt(max_y**2 + max_x**2)
min_r = - max_r
r_one_cell_diff = (max_r - min_r) / r_num_of_bins

for i in okay_lines:
    # print(i[0] * r_one_cell_diff + min_r, i[1] * theta_one_cell_diff + min_theta)
    a3u.draw_line(i[0] * r_one_cell_diff + min_r, i[1] * theta_one_cell_diff + min_theta , *rect_with_lines.shape)
plt.show()








synthetic_image = np.zeros((100, 100))
synthetic_image[10, 10] = 1
synthetic_image[10, 20] = 1
rect_img = synthetic_image
rect_edges = synthetic_image
plt.imshow(rect_edges)
plt.show()



acc = hough_find_lines(rect_edges, r_num_of_bins, theta_num_of_bins, 0)
plt.imshow(acc)
plt.show()
acc_surpresed = nonmaxima_suppression_box(acc)
plt.imshow(acc_surpresed)
plt.show()

rect_with_lines = rect_img.copy()
okay_lines = hough_treshold_lines(acc_surpresed,1)
# print(okay_lines)
plt.imshow(rect_with_lines)




theta_one_cell_diff = np.pi / theta_num_of_bins
min_theta = -np.pi / 2

max_y, max_x = rect_with_lines.shape
max_r = math.sqrt(max_y**2 + max_x**2)
min_r = - max_r
r_one_cell_diff = (max_r - min_r) / r_num_of_bins

for i in okay_lines:
    # print(i[0] * r_one_cell_diff + min_r, i[1] * theta_one_cell_diff + min_theta)
    a3u.draw_line(i[0] * r_one_cell_diff + min_r, i[1] * theta_one_cell_diff + min_theta , *rect_with_lines.shape)
plt.show()








def hough_treshold_lines_top10(parameter_space):
    """takes parameter space with ixs: (r, theta).
    Returns pairs (r, theta) of pixesls with count above the treshold."""

    copied_parameter_space = parameter_space.copy()
    # copied_parameter_space[parameter_space <= treshold] = 0

    # UZ_utils.imshow(copied_parameter_space)

    return_pairs_with_value = list()
    for r in range(copied_parameter_space.shape[0]):
        for theta in range(copied_parameter_space.shape[1]):
            if copied_parameter_space[r, theta] > 0:
                return_pairs_with_value.append(((r, theta), parameter_space[r, theta]))
    

    return_pairs_with_value = sorted(return_pairs_with_value, key=lambda weird_pair: weird_pair[1], reverse=True)
    return_pairs_with_value_top10 = return_pairs_with_value[0:10]
    return_pairs = list()
    for i in return_pairs_with_value_top10:
        return_pairs.append(i[0])
    
    return return_pairs





theta_num_of_bins = 100
r_num_of_bins = 700

sigma = 1
theta = 0.2






rect_img = UZ_utils.imread_gray(".\\images\\rectangle.png")
rect_edges = findedges(rect_img, sigma, theta)
plt.imshow(rect_edges)
plt.show()

acc = hough_find_lines(rect_edges, r_num_of_bins, theta_num_of_bins, 0)
# plt.imshow(acc)
# plt.show()
acc_surpresed = nonmaxima_suppression_box(acc)
# plt.imshow(acc_surpresed)
# plt.show()

rect_with_lines = rect_img.copy()
okay_lines = hough_treshold_lines_top10(acc_surpresed)
# print(okay_lines)
plt.imshow(rect_with_lines)

theta_one_cell_diff = np.pi / theta_num_of_bins
min_theta = -np.pi / 2

max_y, max_x = rect_with_lines.shape
max_r = math.sqrt(max_y**2 + max_x**2)
min_r = - max_r
r_one_cell_diff = (max_r - min_r) / r_num_of_bins

for i in okay_lines:
    # print(i[0] * r_one_cell_diff + min_r, i[1] * theta_one_cell_diff + min_theta)
    a3u.draw_line(i[0] * r_one_cell_diff + min_r, i[1] * theta_one_cell_diff + min_theta , *rect_with_lines.shape)
plt.show()









rect_img = UZ_utils.imread_gray(".\\images\\pier.jpg")
rect_edges = findedges(rect_img, 1.3, theta)
plt.imshow(rect_edges)
plt.show()

acc = hough_find_lines(rect_edges, r_num_of_bins, theta_num_of_bins, 0)
# plt.imshow(acc)
# plt.show()
acc_surpresed = nonmaxima_suppression_box(acc)
# plt.imshow(acc_surpresed)
# plt.show()

rect_with_lines = rect_img.copy()
okay_lines = hough_treshold_lines_top10(acc_surpresed)
# print(okay_lines)
plt.imshow(rect_with_lines)

theta_one_cell_diff = np.pi / theta_num_of_bins
min_theta = -np.pi / 2

max_y, max_x = rect_with_lines.shape
max_r = math.sqrt(max_y**2 + max_x**2)
min_r = - max_r
r_one_cell_diff = (max_r - min_r) / r_num_of_bins

for i in okay_lines:
    # print(i[0] * r_one_cell_diff + min_r, i[1] * theta_one_cell_diff + min_theta)
    a3u.draw_line(i[0] * r_one_cell_diff + min_r, i[1] * theta_one_cell_diff + min_theta , *rect_with_lines.shape)
plt.show()













rect_img = UZ_utils.imread_gray(".\\images\\bricks.jpg")
rect_edges = findedges(rect_img, sigma, theta)
plt.imshow(rect_edges)
plt.show()

acc = hough_find_lines(rect_edges, r_num_of_bins, theta_num_of_bins, 0)
# plt.imshow(acc)
# plt.show()
acc_surpresed = nonmaxima_suppression_box(acc)
# plt.imshow(acc_surpresed)
# plt.show()

rect_with_lines = rect_img.copy()
okay_lines = hough_treshold_lines_top10(acc_surpresed)
# print(okay_lines)
plt.imshow(rect_with_lines)

theta_one_cell_diff = np.pi / theta_num_of_bins
min_theta = -np.pi / 2

max_y, max_x = rect_with_lines.shape
max_r = math.sqrt(max_y**2 + max_x**2)
min_r = - max_r
r_one_cell_diff = (max_r - min_r) / r_num_of_bins

for i in okay_lines:
    # print(i[0] * r_one_cell_diff + min_r, i[1] * theta_one_cell_diff + min_theta)
    a3u.draw_line(i[0] * r_one_cell_diff + min_r, i[1] * theta_one_cell_diff + min_theta , *rect_with_lines.shape)
plt.show()











def hough_find_lines_limited_range(image, r_num_of_bins, theta_num_of_bins, sigma, range_percentage, treshold=0):
    # max r for sure cannot be more than: max_y + max_y, and it cannot be less than -(max_y + max_x)

    _, angles = gradient_magnitude(image, sigma)

    accumulator = np.zeros((r_num_of_bins, theta_num_of_bins))
    
    theta_one_cell_diff = np.pi / theta_num_of_bins
    min_theta = -np.pi / 2

    max_y, max_x = image.shape
    max_r = math.sqrt(max_y**2 + max_x**2)
    min_r = - max_r
    r_one_cell_diff = (max_r - min_r) / r_num_of_bins

    # theta = np.linspace(-np.pi, np.pi, num_of_bins)
    for y_ix in range(image.shape[0]):
        for x_ix in range(image.shape[1]):
            if(image[y_ix, x_ix] == 1):
                curr_angle = angles[y_ix, x_ix]
                if curr_angle > (np.pi / 2):
                    curr_angle -= np.pi
                elif curr_angle < -(np.pi / 2):
                    curr_angle += np.pi
                
                
                theta_min = curr_angle - ((np.pi * range_percentage)/2)
                theta_max = curr_angle + ((np.pi * range_percentage)/2)
                # print(curr_angle, theta_min, theta_max)
                if theta_min < -np.pi/2:
                    theta_min = -np.pi/2
                if theta_max > np.pi/2:
                    theta_max = np.pi/2
                
                theta_ix_min = int((theta_min + min_theta) / theta_one_cell_diff)
                theta_ix_max = int((theta_max + min_theta) / theta_one_cell_diff)

                # print(theta_ix_min, theta_ix_max)


                for theta_ix in range(theta_ix_min, theta_ix_max):
                    theta = theta_ix * theta_one_cell_diff + min_theta

                    r = x_ix * np.cos(theta) + y_ix * np.sin(theta)
                    r = -r
                    r_translated = r + min_r
                    r_ix = int(math.floor(r_translated / r_one_cell_diff))

                    


                    accumulator[r_ix, theta_ix] += 1
    
    return accumulator









sigma = 1
range_procentage = 0.05



# !!!!! this is actually the oneline example

rect_img = UZ_utils.imread_gray(".\\images\\oneline.png")
rect_edges = findedges(rect_img, sigma, 0.2)
plt.imshow(rect_edges)
plt.show()


theta_num_of_bins = 700
r_num_of_bins = theta_num_of_bins

acc = hough_find_lines_limited_range(rect_edges, r_num_of_bins, theta_num_of_bins, sigma, range_procentage, 0)
plt.imshow(acc)
plt.show()
acc_surpresed = nonmaxima_suppression_box(acc)
# plt.imshow(acc_surpresed)
# plt.show()

rect_with_lines = rect_img.copy()
okay_lines = hough_treshold_lines(acc_surpresed, 80)
# print(okay_lines)
plt.imshow(rect_with_lines)




theta_one_cell_diff = np.pi / theta_num_of_bins
min_theta = -np.pi / 2

max_y, max_x = rect_with_lines.shape
max_r = math.sqrt(max_y**2 + max_x**2)
min_r = - max_r
r_one_cell_diff = (max_r - min_r) / r_num_of_bins

for i in okay_lines:
    # print(i[0] * r_one_cell_diff + min_r, i[1] * theta_one_cell_diff + min_theta)
    a3u.draw_line(i[0] * r_one_cell_diff + min_r, i[1] * theta_one_cell_diff + min_theta , *rect_with_lines.shape)
plt.show()











rect_img = UZ_utils.imread_gray(".\\images\\rectangle.png")
rect_edges = findedges(rect_img, sigma, 0.2)
plt.imshow(rect_edges)
plt.show()



acc = hough_find_lines_limited_range(rect_edges, r_num_of_bins, theta_num_of_bins, sigma, range_procentage, 0)
plt.imshow(acc)
plt.show()
acc_surpresed = nonmaxima_suppression_box(acc)
# plt.imshow(acc_surpresed)
# plt.show()

rect_with_lines = rect_img.copy()
okay_lines = hough_treshold_lines(acc_surpresed, 150)
# print(okay_lines)
plt.imshow(rect_with_lines)




theta_one_cell_diff = np.pi / theta_num_of_bins
min_theta = -np.pi / 2

max_y, max_x = rect_with_lines.shape
max_r = math.sqrt(max_y**2 + max_x**2)
min_r = - max_r
r_one_cell_diff = (max_r - min_r) / r_num_of_bins

for i in okay_lines:
    # print(i[0] * r_one_cell_diff + min_r, i[1] * theta_one_cell_diff + min_theta)
    a3u.draw_line(i[0] * r_one_cell_diff + min_r, i[1] * theta_one_cell_diff + min_theta , *rect_with_lines.shape)
plt.show()








def length_of_line(rho, theta, h, w):
    """
    "rho" and "theta": Parameters for the line which will be drawn.
    "h", "w": Height and width of an image.
    """

    c = np.cos(theta)
    s = np.sin(theta)

    xs = []
    ys = []
    if s != 0:
        y = int(rho / s)
        if 0 <= y < h:
            xs.append(0)
            ys.append(y)

        y = int((rho - w * c) / s)
        if 0 <= y < h:
            xs.append(w - 1)
            ys.append(y)
    if c != 0:
        x = int(rho / c)
        if 0 <= x < w:
            xs.append(x)
            ys.append(0)

        x = int((rho - h * s) / c)
        if 0 <= x < w:
            xs.append(x)
            ys.append(h - 1)

    # plt.plot(xs[:2], ys[:2], 'r', linewidth=.7)
    if len(xs) == 2:
        line_length = math.sqrt((xs[0]-xs[1])**2 + (ys[0]-ys[1])**2)
    else:
        line_length = -1
    return line_length



def hough_find_lines_normalised(image, r_num_of_bins, theta_num_of_bins, treshold=0):
    # max r for sure cannot be more than: max_y + max_y, and it cannot be less than -(max_y + max_x)

    accumulator = hough_find_lines(image, r_num_of_bins, theta_num_of_bins, 0)
    
    # Maybe later:
    # Now we just have to divide the count in each cell by the length of the line it represents.
    # Before we do that, we can multiply all amounts by the length of the diagonal - just so we get somewhat more comprehensible numbers.
    lenght_of_diagonal = math.sqrt(image.shape[0]**2 + image.shape[1]**2)

    theta_one_cell_diff = np.pi / theta_num_of_bins
    min_theta = -np.pi / 2

    max_y, max_x = rect_with_lines.shape
    max_r = math.sqrt(max_y**2 + max_x**2)
    min_r = - max_r
    r_one_cell_diff = (max_r - min_r) / r_num_of_bins
    
    for r_ix in range(accumulator.shape[0]):
        for theta_ix in range(accumulator.shape[1]):

            current_r = r_ix * r_one_cell_diff + min_r
            current_theta = theta_ix * theta_one_cell_diff + min_theta

            line_length = length_of_line(current_r, current_theta, image.shape[0], image.shape[1])

            if line_length > 0:
                accumulator[r_ix][theta_ix] /= line_length
            else:
                accumulator[r_ix][theta_ix] = 0





    return accumulator



# def compute_zeros(list_of_r_theta_pairs, h, w):
#     # x = r / cos(theta)
#     for pair in list_of_r_theta_pairs:
#         current_r_ix = pair[0]
#         theta_ix = pair[1]


#         theta_one_cell_diff = np.pi / theta_num_of_bins
#         min_theta = -np.pi / 2

#         max_y, max_x = rect_with_lines.shape
#         max_r = math.sqrt(max_y**2 + max_x**2)
#         min_r = - max_r
#         r_one_cell_diff = (max_r - min_r) / r_num_of_bins

#         current_r = current_r_ix * r_one_cell_diff + min_r
#         theta = theta_one_cell_diff * current_r + min_theta

#         x_when_y_is_zero = current_r / math.cos(theta)

#         # x = (r - image.shape[0] sin(theta)) / cos(theta)
#         x_when_y_is_max = (current_r - h * math.sin(theta) )/ math.cos(theta)

#         # y = r / sin(theta)
#         y_when_x_is_zero = current_r / math.sin(theta)

#         # y = (r - image.shape[1] cos(theta)) / sin(theta)
#         y_when_x_is_max = (current_r - w * math.cos(theta) )/ math.sin(theta)

#         print("x_when_y_is_zero, x_when_y_is_max , y_when_x_is_zero, y_when_x_is_max")
#         print(x_when_y_is_zero, x_when_y_is_max , y_when_x_is_zero, y_when_x_is_max)





r_num_of_bins, theta_num_of_bins = (500, 1000)


rect_img = UZ_utils.imread_gray(".\\images\\rectangle.png")
rect_edges = findedges(rect_img, 1, 0.2)
# print(rect_edges)
plt.imshow(rect_edges)
plt.show()



acc = hough_find_lines_normalised(rect_edges, r_num_of_bins, theta_num_of_bins, 0)
# plt.imshow(acc)
# plt.show()
acc_surpresed = nonmaxima_suppression_box(acc)
# plt.imshow(acc_surpresed)
# plt.show()



rect_with_lines = rect_img.copy()
okay_lines = hough_treshold_lines_top10(acc_surpresed)
# print(okay_lines)
plt.imshow(rect_img)






theta_one_cell_diff = np.pi / theta_num_of_bins
min_theta = -np.pi / 2

max_y, max_x = rect_with_lines.shape
max_r = math.sqrt(max_y**2 + max_x**2)
min_r = - max_r
r_one_cell_diff = (max_r - min_r) / r_num_of_bins

for i in okay_lines:
    # print(i[0] * r_one_cell_diff + min_r, i[1] * theta_one_cell_diff + min_theta)
    a3u.draw_line(i[0] * r_one_cell_diff + min_r, i[1] * theta_one_cell_diff + min_theta , *rect_with_lines.shape)
plt.show()

# print(rect_edges.shape)
# print(compute_zeros(okay_lines, rect_edges.shape[0], rect_edges.shape[1]))








synthetic_image = np.zeros((100, 500))
synthetic_image[25:75, 200:300] = 1

rect_img = synthetic_image
rect_edges = findedges(rect_img, 1, 0.2)
# print(rect_edges)
plt.imshow(rect_edges)
plt.show()



acc = hough_find_lines_normalised(rect_edges, r_num_of_bins, theta_num_of_bins, 0)
# plt.imshow(acc)
# plt.show()
acc_surpresed = nonmaxima_suppression_box(acc)
# plt.imshow(acc_surpresed)
# plt.show()



rect_with_lines = rect_img.copy()
okay_lines = hough_treshold_lines_top10(acc_surpresed)
# print(okay_lines)
plt.imshow(rect_img)






theta_one_cell_diff = np.pi / theta_num_of_bins
min_theta = -np.pi / 2

max_y, max_x = rect_with_lines.shape
max_r = math.sqrt(max_y**2 + max_x**2)
min_r = - max_r
r_one_cell_diff = (max_r - min_r) / r_num_of_bins

for i in okay_lines:
    # print(i[0] * r_one_cell_diff + min_r, i[1] * theta_one_cell_diff + min_theta)
    a3u.draw_line(i[0] * r_one_cell_diff + min_r, i[1] * theta_one_cell_diff + min_theta , *rect_with_lines.shape)
plt.show()

# print(rect_edges.shape)
# print(compute_zeros(okay_lines, rect_edges.shape[0], rect_edges.shape[1]))







