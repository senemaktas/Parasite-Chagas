import cv2
import numpy as np
import os


def morph_operations(img):
    # can use different kernels with this rather than opening
    erosion_kernel = np.ones((10, 15), np.uint8)
    dilation_kernel = np.ones((10, 10), np.uint8)

    # The first parameter is the original image,kernel is the matrix with which image is convolved and
    # third parameter is the number of iterations, which will determine how much you want to erode/dilate
    # a given image.
    frame_erode = cv2.erode(img, erosion_kernel, iterations=1)
    out_img = cv2.dilate(frame_erode, dilation_kernel, iterations=1)
    # opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return out_img


def img_sharpening(img):
    # create a sharpening kernel
    sharpen_filter1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpen_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # applying kernels to the input image to get the sharpened image
    sharp_image = cv2.filter2D(img, -1, sharpen_filter)
    # Sharpen the image using the Laplacian operator # sharpened_image2 = cv2.Laplacian(self.img, cv2.CV_64F)
    return sharp_image


def clahe_equalization(img):
    c_limit = 1.5
    grid = 7
    img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    # clipLimit : This is the threshold for contrast limiting
    # tileGridSize : Divides the input image into M x N tiles and then
    # applies histogram equalization to each local tile
    clahe = cv2.createCLAHE(clipLimit=c_limit, tileGridSize=(grid, grid))
    # 0 to 'L' channel, 1 to 'a' channel, and 2 to 'b' channel
    img[:, :, 0] = clahe.apply(img[:, :, 0])
    img = cv2.cvtColor(img, cv2.COLOR_Lab2RGB)
    return img


input_video_path = "./chagas_out/chagas33_preprocess_out_cleaned_out_flow_dense.avi"
out_put_folder = "./chagas_out/"

vidcap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)
fps = vidcap.get(cv2.CAP_PROP_FPS)
video_width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
video_height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
frame_num = 0

out_vid_name = os.path.basename(input_video_path)
out_vid_name = out_vid_name.split('.')[0]

# ****************************************************************
# convert grayscale - sharpen - save as avi as grayscale video
# ****************************************************************
preprocess_out_path = str(out_put_folder) + str(out_vid_name) + '_THRESH_TOZERO_INV.avi'
out = cv2.VideoWriter(preprocess_out_path, cv2.VideoWriter_fourcc(*'MPEG'),
                      fps, (int(video_width), int(video_height)), 0)  # MPEG DIVX 0 for grayscala
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (image.shape[1], image.shape[0]), 0)

while vidcap.isOpened():
    ret, frame = vidcap.read()
    if ret is not True:
        break
    if ret is True:
        frame_num += 1
        dh, dw, _ = frame.shape
        # print('Frame #: ', frame_num)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # techniques on the input image all pixels value above 120 will be set to 255
        ret, thresh1 = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
        ret, thresh2 = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY_INV)
        ret, thresh3 = cv2.threshold(img, 80, 255, cv2.THRESH_TRUNC)
        ret, thresh4 = cv2.threshold(img, 80, 255, cv2.THRESH_TOZERO)
        ret, thresh5 = cv2.threshold(img, 80, 255, cv2.THRESH_TOZERO_INV)

        # the window showing output images
        # with the corresponding thresholding
        # techniques applied to the input images
        cv2.imshow('Binary Threshold', thresh1)
        # cv2.imshow('Binary Threshold Inverted', thresh2)
        # cv2.imshow('Truncated Threshold', thresh3)
        # cv2.imshow('Set to 0', thresh4)
        # cv2.imshow('Set to 0 Inverted', thresh5)
        # cv2.waitKey(0)

        out.write(thresh5)
        if frame_num == 50:
            break

out.release()
vidcap.release()

