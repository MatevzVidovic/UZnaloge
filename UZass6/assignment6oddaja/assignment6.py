import numpy as np
import a6_utils as a6u
# import UZ_utils
import matplotlib.pyplot as plt
import cv2
import math
import os
import random
import UZ_utils


plt.rcParams['figure.figsize'] = [8, 4]






def e1b():
    ptsTrans = np.loadtxt(".\data\points.txt")
    pts = ptsTrans.T.copy()

    # pts = np.append(pts, [[-2], [10]], axis=1)

    # print("pts")
    # print(pts)


    meanVec = np.mean(pts, axis=1).reshape(2,1)
    # print(meanVec)

    X = pts - meanVec
    # print(X)

    numOfPts = pts.shape[1]
    C = 1 / (numOfPts - 1) * (X @ X.T)
    # print(C)

    U, S, _ = np.linalg.svd(C)

    plt.scatter(pts[0,:], pts[1,:])
    a6u.drawEllipse(meanVec, C)

    # print("U")
    # print(U)


    eigenvec1Endpoint = meanVec + U[:,0].reshape(-1,1) * np.sqrt(S[0])
    eigenvec2Endpoint = meanVec + U[:,1].reshape(-1,1) * np.sqrt(S[1])

    # print(U[:,0])
    # print(S[0])
    # print("eigenvec1Endpoint")
    # print(eigenvec1Endpoint)


    plt.plot([meanVec[0], eigenvec1Endpoint[0]], [meanVec[1], eigenvec1Endpoint[1]], color="red")
    plt.plot([meanVec[0], eigenvec2Endpoint[0]], [meanVec[1], eigenvec2Endpoint[1]], color="green")

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    plt.show()


e1b()














def e1d():
    ptsTrans = np.loadtxt(".\data\points.txt")
    pts = ptsTrans.T.copy()

    # pts = np.append(pts, [[-2], [10]], axis=1)

    # print("pts")
    # print(pts)


    meanVec = np.mean(pts, axis=1).reshape(2,1)
    # print(meanVec)

    X = pts - meanVec
    # print(X)

    numOfPts = pts.shape[1]
    C = 1 / (numOfPts - 1) * (X @ X.T)
    # print(C)

    U, S, _ = np.linalg.svd(C)

    S = S / np.sum(S)
    S = np.cumsum(S)
    print(S)

    plt.bar([1,2], S)
    plt.show()



e1d()
















def e1e():

    ptsTrans = np.loadtxt(".\data\points.txt")
    pts = ptsTrans.T.copy()

    # pts = np.append(pts, [[-2], [10]], axis=1)

    # print("pts")
    # print(pts)


    meanVec = np.mean(pts, axis=1).reshape(2,1)
    # print(meanVec)

    X = pts - meanVec
    # print(X)

    numOfPts = pts.shape[1]
    C = 1 / (numOfPts - 1) * (X @ X.T)
    # print(C)

    U, S, _ = np.linalg.svd(C)


    pts = U.T @ X
    pts[1,:] = 0
    pts = U @ pts + meanVec
    



    plt.scatter(pts[0,:], pts[1,:])
    a6u.drawEllipse(meanVec, C)

    # print("U")
    # print(U)


    eigenvec1Endpoint = meanVec + U[:,0].reshape(-1,1) * np.sqrt(S[0])
    eigenvec2Endpoint = meanVec + U[:,1].reshape(-1,1) * np.sqrt(S[1])

    # print(U[:,0])
    # print(S[0])
    # print("eigenvec1Endpoint")
    # print(eigenvec1Endpoint)


    plt.plot([meanVec[0], eigenvec1Endpoint[0]], [meanVec[1], eigenvec1Endpoint[1]], color="red")
    plt.plot([meanVec[0], eigenvec2Endpoint[0]], [meanVec[1], eigenvec2Endpoint[1]], color="green")

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    plt.show()

e1e()


















def e1f():

    ptsTrans = np.loadtxt(".\data\points.txt")
    pts = ptsTrans.T.copy()

    # pts = np.append(pts, [[-2], [10]], axis=1)

    # print("pts")
    # print(pts)

    
    newPoint = [[6],[6]]

    distances = np.linalg.norm(pts - newPoint, axis=0)
    print("distances")
    print(distances)
    ixOfMin = np.argmin(distances)
    print("ixOfMin")
    print(ixOfMin)






    meanVec = np.mean(pts, axis=1).reshape(2,1)
    # print(meanVec)

    X = pts - meanVec
    # print(X)

    numOfPts = pts.shape[1]
    C = 1 / (numOfPts - 1) * (X @ X.T)
    # print(C)

    U, S, _ = np.linalg.svd(C)







    plt.scatter(pts[0,:], pts[1,:])
    plt.scatter(newPoint[0], newPoint[1], c="red")
    a6u.drawEllipse(meanVec, C)

    # print("U")
    # print(U)


    eigenvec1Endpoint = meanVec + U[:,0].reshape(-1,1) * np.sqrt(S[0])
    eigenvec2Endpoint = meanVec + U[:,1].reshape(-1,1) * np.sqrt(S[1])

    # print(U[:,0])
    # print(S[0])
    # print("eigenvec1Endpoint")
    # print(eigenvec1Endpoint)


    plt.plot([meanVec[0], eigenvec1Endpoint[0]], [meanVec[1], eigenvec1Endpoint[1]], color="red")
    plt.plot([meanVec[0], eigenvec2Endpoint[0]], [meanVec[1], eigenvec2Endpoint[1]], color="green")

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    plt.show()









    
    




    pts = U.T @ X
    pts[1,:] = 0
    pts = U @ pts + meanVec


    newPoint = U.T @ (newPoint - meanVec)
    newPoint[1] = 0
    newPoint = U @ newPoint + meanVec

    distances = np.linalg.norm(pts - newPoint, axis=0)
    print("distances")
    print(distances)
    ixOfMin = np.argmin(distances)
    print("ixOfMin")
    print(ixOfMin)
    






    plt.scatter(pts[0,:], pts[1,:])
    plt.scatter(newPoint[0], newPoint[1], c="red")
    a6u.drawEllipse(meanVec, C)

    # print("U")
    # print(U)


    eigenvec1Endpoint = meanVec + U[:,0].reshape(-1,1) * np.sqrt(S[0])
    eigenvec2Endpoint = meanVec + U[:,1].reshape(-1,1) * np.sqrt(S[1])

    # print(U[:,0])
    # print(S[0])
    # print("eigenvec1Endpoint")
    # print(eigenvec1Endpoint)


    plt.plot([meanVec[0], eigenvec1Endpoint[0]], [meanVec[1], eigenvec1Endpoint[1]], color="red")
    plt.plot([meanVec[0], eigenvec2Endpoint[0]], [meanVec[1], eigenvec2Endpoint[1]], color="green")

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    plt.show()

e1f()




















def e2a():
    ptsTrans = np.loadtxt(".\data\points.txt")
    pts = ptsTrans.T.copy()

    # pts = np.append(pts, [[-2], [10]], axis=1)

    # print("pts")
    # print(pts)


    meanVec = np.mean(pts, axis=1).reshape(2,1)
    # print(meanVec)

    X = pts - meanVec
    # print(X)

    numOfPts = pts.shape[1]
    C = 1 / (numOfPts - 1) * (X.T @ X)
    # print(C)

    U, S, _ = np.linalg.svd(C)

    print("U before multiplication")
    print(U)
    print("S")
    print(S)


    U = (X @ U) * ( (S*(numOfPts-1))**(-1/2))



    # plt.scatter(pts[0,:], pts[1,:])
    # a6u.drawEllipse(meanVec, C)

    print("U")
    print(U)


    # eigenvec1Endpoint = meanVec + U[:,0].reshape(-1,1) * np.sqrt(S[0])
    # eigenvec2Endpoint = meanVec + U[:,1].reshape(-1,1) * np.sqrt(S[1])

    # print(U[:,0])
    # print(S[0])
    # print("eigenvec1Endpoint")
    # print(eigenvec1Endpoint)


    # plt.plot([meanVec[0], eigenvec1Endpoint[0]], [meanVec[1], eigenvec1Endpoint[1]], color="red")
    # plt.plot([meanVec[0], eigenvec2Endpoint[0]], [meanVec[1], eigenvec2Endpoint[1]], color="green")

    # ax = plt.gca()
    # ax.set_aspect('equal', adjustable='box')

    # plt.show()


e2a()



















def e2b():
    ptsTrans = np.loadtxt(".\data\points.txt")
    pts = ptsTrans.T.copy()

    # pts = np.append(pts, [[-2], [10]], axis=1)

    print("pts before:")
    print(pts)


    meanVec = np.mean(pts, axis=1).reshape(2,1)
    # print(meanVec)

    X = pts - meanVec
    # print(X)

    numOfPts = pts.shape[1]
    C = 1 / (numOfPts - 1) * (X.T @ X)
    # print(C)

    U, S, _ = np.linalg.svd(C)

    U = (X @ U) * ( (S*(numOfPts-1))**(-1/2))



    # plt.scatter(pts[0,:], pts[1,:])
    # a6u.drawEllipse(meanVec, C)

    # print("U")
    # print(U)


    pts = U.T @ X
    # pts[1,:] = 0
    pts = U @ pts + meanVec

    print("pts after:")
    print(pts)


    


    # eigenvec1Endpoint = meanVec + U[:,0].reshape(-1,1) * np.sqrt(S[0])
    # eigenvec2Endpoint = meanVec + U[:,1].reshape(-1,1) * np.sqrt(S[1])

    # print(U[:,0])
    # print(S[0])
    # print("eigenvec1Endpoint")
    # print(eigenvec1Endpoint)


    # plt.plot([meanVec[0], eigenvec1Endpoint[0]], [meanVec[1], eigenvec1Endpoint[1]], color="red")
    # plt.plot([meanVec[0], eigenvec2Endpoint[0]], [meanVec[1], eigenvec2Endpoint[1]], color="green")

    # ax = plt.gca()
    # ax.set_aspect('equal', adjustable='box')

    # plt.show()


e2b()
















def getFacesMatrix():

    base_path = ".\\data\\faces\\1"
    dir_list = os.listdir(base_path)
    # print(dir_list)

    images = []
    for name in dir_list:
        read_image = UZ_utils.imread_gray(base_path + "\\" + name)
        # print(read_image.shape)
        reshaped_img = read_image.reshape(-1)
        images.append(reshaped_img)


    image_length = images[0].shape[0]
    M = np.zeros((image_length, len(images)))
    for ix in range(len(images)):
        M[:, ix] = images[ix]

    # print(M)

    return M



getFacesMatrix().shape




















def dualPCA(facesMatrix):

    pts = facesMatrix

    # print("pts")
    # print(pts)


    meanVec = np.mean(pts, axis=1).reshape(-1,1)
    # print(meanVec)

    X = pts - meanVec
    # print(X)

    numOfPts = pts.shape[1]
    C = 1 / (numOfPts - 1) * (X.T @ X)
    # print(C)

    U, S, _ = np.linalg.svd(C)

    S = S + np.ones(S.shape[0]) * 1e-15

    U = (X @ U) * ( (S*(numOfPts-1))**(-1/2))



    # plt.scatter(pts[0,:], pts[1,:])
    # a6u.drawEllipse(meanVec, C)

    # print("U")
    # print(U)

    return U, meanVec

dualPCA(getFacesMatrix())














def e3b():

    M = getFacesMatrix()
    U, meanVec = dualPCA(M)

    for i in range(5):
        plt.subplot(1,5, i+1)
        img = U[:, i].reshape(96, 84)
        # print(img.shape)
        plt.imshow(img, cmap="gray")
    
    plt.show()
    

    # print("U.shape")
    # print(U.shape)


    first_image = UZ_utils.imread_gray(".\\data\\faces\\1\\001.png").reshape(-1, 1)
    pcaSpace_img = U.T @ (first_image - meanVec)
    imgSpace_img = U @ pcaSpace_img + meanVec

    diff = np.sum((first_image - imgSpace_img)**2)
    print("euclidian distance between initial image and reconstructed image")
    print(diff)

    # plt.imshow(first_image.reshape(96, 84), cmap="gray")
    # plt.show()
    # plt.imshow(imgSpace_img.reshape(96, 84), cmap="gray")
    # plt.show()

    img_space_change = first_image.copy()
    img_space_change[4076] = 0

    pcaSpace_img[0] = 0
    imgSpace_img = U @ pcaSpace_img + meanVec

    plt.imshow(img_space_change.reshape(96, 84), cmap="gray")
    plt.show()
    plt.imshow(imgSpace_img.reshape(96, 84), cmap="gray")
    plt.show()

    diff_vec = np.abs(first_image - img_space_change)
    num_of_diffs = np.sum(diff_vec > 1e-15)
    print("num_of_diffs img space change")
    print(num_of_diffs)


    diff_vec = np.abs(first_image - imgSpace_img)
    num_of_diffs = np.sum(diff_vec > 10e-15)
    print("num_of_diffs pca space change")
    print(num_of_diffs)






e3b()















def e3c():

    M = getFacesMatrix()
    U, meanVec = dualPCA(M)


    first_image = UZ_utils.imread_gray(".\\data\\faces\\1\\001.png").reshape(-1, 1)
    pcaSpace_img = U.T @ (first_image - meanVec)

    for i in range(6):
        reduct_num = (2**i)

        pcaSpace_img_reduced = pcaSpace_img.copy()
        pcaSpace_img_reduced[reduct_num:] = 0

        imgSpace_img = U @ pcaSpace_img_reduced + meanVec



        plt.subplot(1,6,6-i)
        plt.title(reduct_num)
        plt.imshow(imgSpace_img.reshape(96, 84), cmap="gray")

    plt.show()



e3c()






















