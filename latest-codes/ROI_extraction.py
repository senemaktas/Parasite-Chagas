import cv2
import numpy as np
from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

class BackgroundSegmentation:

    def __init__(self, frame):
        self.frame = frame

    def back_segment(self):
        # Arkaplan çıkarılır(iki görüntü birbirinden de denilebilir)
        # Eğer detectShadows True ise gölgeler gösterilir, yukarıda oluşturulan maske görüntüye uygulanır
        fgmask = fgbg.apply(self.frame)
        # median = cv2.medianBlur(fgmask, 1)
        # (contours, hierarchy) = cv2.findContours(median.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (contours, hierarchy) = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        send_track_bbox = []
        for c in contours:
            if cv2.contourArea(c) < 200:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            x, y, w, h = [0 if (a < 0) else int(a) for a in [x, y, w, h]]

            center_coordinates = (int(x + (w / 2)), int(y + (h / 2)))
            cv2.circle(self.frame, center_coordinates, radius=5, color=(0, 0, 255), thickness=-1)

            # cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            # bbs = tuple(([x, y, w, h], 0.9, 1))
            bbs = [[x, y, x + w, y + h], 0.9, 1]  # bytetrack
            send_track_bbox.append(bbs)
        return send_track_bbox, self.frame


class SkimageBlobDetections:

    def __init__(self, original_frame, flow_frame):
        self.original_frame = original_frame
        self.flow_frame = flow_frame
        self.blob_sequences = 0

    def parameters_def(self):
        pass

    def blob_log_detection(self):
        # https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html
        # https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_doh
        image_gray = rgb2gray(self.flow_frame)

        # -------------
        # blob log
        # -------------
        blobs_log = blob_log(image_gray.astype(float), max_sigma=15, min_sigma=1, num_sigma=60, threshold=.1)  #,min_sigma=0.4
        # Compute radii in the 3rd column.
        blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

        # -------------
        # blob dog
        # https://scikit-image.org/docs/stable/user_guide/data_types.html
        # -------------
        blobs_dog = blob_dog(image_gray.astype(float), max_sigma=30, min_sigma=2, threshold=.1, overlap=0.7)
        blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

        # -------------
        # blob doh
        # -------------
        blobs_doh = blob_doh(image_gray.astype(float), max_sigma=5, min_sigma=1, threshold=.01, num_sigma=20)

        # -------------
        # draw blobs results
        # -------------
        blobs_list = [blobs_log, blobs_dog, blobs_doh]
        titles = ['Laplacian of Gaussian', 'Difference of Gaussian', 'Determinant of Hessian']
        blob_sequences = zip(blobs_list, titles)

        img0 = cv2.copyMakeBorder(self.original_frame.copy(), top=100, bottom=10, left=10, right=10,
                                  borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
        img1 = cv2.copyMakeBorder(self.original_frame.copy(), top=100, bottom=10, left=10, right=10,
                                  borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
        img2 = cv2.copyMakeBorder(self.original_frame.copy(), top=100, bottom=10, left=10, right=10,
                                  borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
        name_img = {"img0": img0, "img1": img1, "img2": img2}

        img_list = []
        for idx, (blobs, title) in enumerate(blob_sequences):
            img = name_img["img" + str(idx)]
            cv2.putText(img, str(titles[idx]), (25, 85), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                        color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(img, "number of blob : " + str(len(blobs)), (45, 45), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                        color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            for blob in blobs:
                y, x, r = blob
                cv2.circle(img, (int(x), int(y)), radius=int(r + 5), color=(0, 0, 255), thickness=3)
            img_list.append(img)
        return blob_sequences, blobs_log, blobs_dog, blobs_doh, img_list

