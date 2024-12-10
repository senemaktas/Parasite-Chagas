import os
import cv2
import numpy as np


class ChagasPreProcess:

    def __init__(self, input_video_path, out_put_folder, frame_num):
        self.input_video_path = input_video_path
        self.out_put_folder = out_put_folder
        self.frame_num = frame_num
        self.c_limit = 1.5
        self.grid = 7

    @staticmethod
    def img_sharpening(img):
        # create a sharpening kernel
        sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        edge_det_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        # applying kernels to the input image to get the sharpened image
        sharp_image = cv2.filter2D(img, -1, sharpen_filter)
        # Sharpen the image using the Laplacian operator # sharpened_image2 = cv2.Laplacian(self.img, cv2.CV_64F)
        return sharp_image

    def clahe_equalization(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        # clipLimit : This is the threshold for contrast limiting
        # tileGridSize : Divides the input image into M x N tiles and then
        # applies histogram equalization to each local tile
        clahe = cv2.createCLAHE(clipLimit=self.c_limit, tileGridSize=(self.grid, self.grid))
        # 0 to 'L' channel, 1 to 'a' channel, and 2 to 'b' channel
        img[:, :, 0] = clahe.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_Lab2RGB)
        return img

    @staticmethod
    def img_grayscale(img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray_img

    def img_hsv_color_space(self):
        pass

    def return_preprocessed_img(self, img):
        sharpen_img = self.img_sharpening(img)
        # clahe_img = self.clahe_equalization(sharpen_img)
        grayscale_img = self.img_grayscale(sharpen_img)
        return grayscale_img

    def return_preprocess(self):
        vidcap = cv2.VideoCapture(self.input_video_path, cv2.CAP_FFMPEG)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        video_width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
        video_height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        frame_num = 0

        out_vid_name = os.path.basename(self.input_video_path)
        out_vid_name = out_vid_name.split('.')[0]

        # ****************************************************************
        # convert grayscale - sharpen - save as avi as grayscale video
        # ****************************************************************
        preprocess_out_path = str(self.out_put_folder) + str(out_vid_name) + '_preprocess_out.avi'
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
                # -------------------------------------------------------------
                # preprocess the images and save the video as avi file
                # -------------------------------------------------------------
                grayscale_img = self.return_preprocessed_img(frame)

                out.write(grayscale_img)
                if frame_num == self.frame_num:
                    break

        out.release()
        vidcap.release()

        return preprocess_out_path
