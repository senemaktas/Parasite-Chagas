import os
import cv2
import numpy as np
from scipy import ndimage


# https://www.geeksforgeeks.org/python-opencv-background-subtraction/
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)


class BackgroundSegmentation:

    # def __init__(self, frame, img_name):
    def __init__(self, frame):
        self.frame = frame
        # self.img_name = img_name

    @staticmethod
    def morph_operations(img):
        # can use different kernels with this rather than opening
        erosion_kernel = np.ones((8, 8), np.uint8)
        dilation_kernel = np.ones((12, 12), np.uint8)

        # The first parameter is the original image,kernel is the matrix with which image is convolved and
        # third parameter is the number of iterations, which will determine how much you want to erode/dilate
        # a given image.
        frame_erode = cv2.erode(img, erosion_kernel, iterations=1)
        out_img = cv2.dilate(frame_erode, dilation_kernel, iterations=1)
        # opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return out_img

    # im = np.zeros((20, 20))
    # im[5:-5, 5:-5] = 1
    # im = ndimage.distance_transform_bf(im)
    # im_noise = im + 0.2 * np.random.randn(*im.shape)
    # im_med = ndimage.median_filter(im_noise, 3)

    @staticmethod
    def yolobbox2bbox(x, y, w, h):
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2
        return x1, y1, x2, y2

    def roi_boxes(self):

        current_frame = self.morph_operations(self.frame)

        # Arkaplan çıkarılır(iki görüntü birbirinden de denilebilir)
        # Eğer detectShadows True ise gölgeler gösterilir, yukarıda oluşturulan maske görüntüye uygulanır
        # fgmask = fgbg.apply(current_frame)
        # median = cv2.medianBlur(fgmask, 1)
        # (contours, hierarchy) = cv2.findContours(median.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gray_img = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        (contours, hierarchy) = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        # (contours, hierarchy) = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        initial_img = np.zeros((1080, 1920, 3), dtype=np.uint8)

        send_track_points = []
        send_track_bbox = []
        for c in contours:
            if cv2.contourArea(c) < 100:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            x, y, w, h = [0 if (a < 0) else int(a) for a in [x, y, w, h]]

            center_coordinates = (int(x + (w / 2)), int(y + (h / 2)))
            send_track_points.append(list(center_coordinates))
            # cv2.circle(initial_img, center_coordinates, radius=5, color=(0, 0, 255), thickness=-1)
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            # bbs = tuple(([x, y, w, h], 0.9, 1))
            # bbs = [[x, y, x + w, y + h], 0.9, 1]  # bytetrack
            x1, y1, x2, y2 = self.yolobbox2bbox(x, y, w, h)
            bbs = [[x1, y1, x2, y2], 0.9, 1]  # deepsort

            send_track_bbox.append(bbs)

        # cv2.imshow("erosion_dil", current_frame)
        # cv2.waitKey(0)
        # cv2.imshow("initial_img", self.frame)
        # current_frame = cv2.copyMakeBorder(current_frame, top=10, bottom=10, left=10, right=10,
        #                                 borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
        # cv2.imwrite("./fastsam_background_comparison/output/background_" + str(self.img_name) + ".png", current_frame)
        # cv2.waitKey(0)

        return send_track_bbox, send_track_points, initial_img  # self.frame


# if __name__ == "__main__":
#     input_folder = "./fastsam_background_comparison/input"
#     for img_path in os.listdir(input_folder):
#         img_name = os.path.basename(img_path).split('.')[0]
#         img_full_path = input_folder + "/" + img_path
#         img = cv2.imread(img_full_path)
#         background_segmentation_result = BackgroundSegmentation(img, img_name).roi_boxes()
