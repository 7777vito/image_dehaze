import cv2
import math
import numpy as np


def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000), 1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def dark_channel_prior(im, size=3):
    """ Compute the dark channel prior for an image based on the updated formula. """
    h, w, _ = im.shape
    padded_im = cv2.copyMakeBorder(
        im, size//2, size//2, size//2, size//2, cv2.BORDER_REFLECT)

    min_channel = np.min(im, axis=2)
    dark_channel = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            local_patch = min_channel[i:i+size, j:j+size]
            dark_channel[i, j] = np.min(local_patch)

    return dark_channel


def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind]/A[0, ind]

    # transmission = 1 - omega*dark_channel_prior(im3, sz)
    transmission = 1 - omega*DarkChannel(im3, sz)
    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im*p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(im*im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a*im + mean_b
    return q


# def TransmissionRefine(im, et):
#     gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     gray = np.float64(gray)/255
#     r = 60
#     eps = 0.0001
#     t = Guidedfilter(gray, et, r, eps)

#     return t

def TransmissionRefine(im, et, edge_threshold=0.1):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges = edges / 255.0  # Normalize edges to [0, 1]

    # Initialize transmission refinement
    t_refined = et.copy()
    rows, cols = et.shape

    # low-pass filter kernel
    low_pass_kernel = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]]) / 16.0

    # Apply low-pass filter
    low_pass_filter = cv2.filter2D(et, -1, low_pass_kernel)
    # show the low-pass filter image

    # mean filter
    mean_filter = cv2.blur(et, (3, 3))
    # show the mean filter image
    for i in range(rows):
        for j in range(cols):
            if edges[i, j] > edge_threshold:
                # Apply low-pass filter for edge regions
                # to [0, 1]
                t_refined[i, j] = low_pass_filter[i, j]
            else:
                # Apply mean filter for non-edge regions
                t_refined[i, j] = mean_filter[i, j]

    return t_refined


def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind]-A[0, ind])/t + A[0, ind]

    return res


if __name__ == '__main__':
    # read all png in path
    path = "./image"
    output = "./output_origin"
    import os
    # import sys
    for fn in os.listdir(path):

        # try:
        #     fn = sys.argv[1]
        # except:
        #     fn = './image/3.png'

        # def nothing(*argv):
        #     pass
        if "_" in fn:
            continue
        src = cv2.imread(path+"/"+fn)
        if src is None:
            continue

        I = src.astype('float64')/255

        dark = DarkChannel(I, 15)
        A = AtmLight(I, dark)
        te = TransmissionEstimate(I, A, 3)
        t = TransmissionRefine(src, te)
        J = Recover(I, t, A, 0.1)

        # cv2.imshow("dark", dark)
        # cv2.imshow("t", t)
        # cv2.imshow('I', src)
        # cv2.imshow('J', J)
        # cv2.imwrite(output+"/"+fn, J*255)
        cv2.imwrite(output+"/"+fn, J*255)
        print(fn)
        # cv2.waitKey()
        cv2.destroyAllWindows()
