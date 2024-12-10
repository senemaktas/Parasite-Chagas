import os
import cv2
import numpy as np
from itertools import chain
from skimage.metrics import structural_similarity


class SimilarityLabelClass:

    def __init__(self, blob_roi_image, labeled_image_folder):
        self.blob_roi_image = blob_roi_image
        self.labeled_image_folder = labeled_image_folder

    @staticmethod
    def structural_similarity_score_diff(blob_roi_image, labeled_image):
        # Convert images to grayscale
        blob_roi_image_gray = cv2.cvtColor(blob_roi_image, cv2.COLOR_BGR2GRAY)
        labeled_image_gray = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2GRAY)
        # Compute SSIM between two images
        score, diff = structural_similarity(blob_roi_image_gray, labeled_image_gray, full=True,
                                            gaussian_weights=True, sigma=3, use_sample_covariance=False)
        return round(score, 3), diff

    @staticmethod
    def visualize_structural_similarity(blob_roi_image, labeled_image, image_path, score, diff):

        # The diff image contains the actual image differences between the two images and is represented as a
        # floating point data type in the range [0,1] so we must convert the array to 8-bit unsigned integers
        # in the range [0,255] before we can use it with OpenCV
        diff = (diff * 255).astype("uint8")
        diff_three_channel = np.stack((diff,) * 3, axis=-1)

        # Threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        mask = np.zeros(blob_roi_image.shape, dtype='uint8')
        filled_after = labeled_image.copy()

        for c in contours:
            area = cv2.contourArea(c)
            if area > 40:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(blob_roi_image, (x, y), (x + w, y + h), (36,255,12), 2)
                cv2.rectangle(labeled_image, (x, y), (x + w, y + h), (36,255,12), 2)
                cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
                cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

        blob_roi_image = cv2.resize(blob_roi_image, (150, 150),interpolation=cv2.INTER_AREA)
        labeled_image = cv2.resize(labeled_image, (150, 150),interpolation=cv2.INTER_AREA)
        diff_three_channel = cv2.resize(diff_three_channel, (150, 150),interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (150, 150),interpolation=cv2.INTER_AREA)
        filled_after = cv2.resize(filled_after, (150, 150),interpolation=cv2.INTER_AREA)

        image_structural_similarity = np.concatenate((blob_roi_image, labeled_image, diff_three_channel, mask, filled_after), axis=1)
        image_structural_similarity = cv2.copyMakeBorder(image_structural_similarity, top=60, bottom=10, left=10,
                                                         right=10, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.putText(image_structural_similarity, "Similar Image Name: " + str(image_path) + '  Similarity Score:' + str(score) ,
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        cv2.putText(image_structural_similarity, 'blob_roi_image - labeled_image - diff_three_channel - mask - filled_after',
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        return image_structural_similarity

    def bring_n_number_most_similar_label_result(self):
        n_number_most_similar = 5
        n_most_similar_score_result = []

        i = 0
        for image_path in os.listdir(self.labeled_image_folder):
            input_path = os.path.join(self.labeled_image_folder, image_path)
            labeled_image = cv2.imread(input_path)  # each image on the label set
            new_similarity_score, diff = self.structural_similarity_score_diff(self.blob_roi_image.copy(),
                                                                               labeled_image.copy())
            # Filling with values up to the first n_most_similar_score_result.
            if len(n_most_similar_score_result) < n_number_most_similar:
                n_most_similar_score_result.append({"similarity_score": new_similarity_score, "label_image_path": input_path})
            i += 1

            # If n_number_most_similar == i (means we filled first n_number_most_similar value then compare others),
            # compare the new value with the values inside n_number_most_similar,
            # if the score is more, take it, otherwise skip it.
            if i > n_number_most_similar:
                # get the min score value from dictionary to compare with new score
                saved_scores = list(chain([item["similarity_score"] for item in n_most_similar_score_result]))
                saved_minimum_score_index = saved_scores.index(min(saved_scores))

                if new_similarity_score > min(saved_scores):
                    n_most_similar_score_result[saved_minimum_score_index].update({"similarity_score": new_similarity_score,
                                                                                   "label_image_path": input_path})

        # -------------------------------------------------
        # ***** Get the value with the maximum score  *****
        # -------------------------------------------------
        result_saved_scores = list(chain([item["similarity_score"] for item in n_most_similar_score_result]))
        result_saved_maximum_score_index = result_saved_scores.index(max(result_saved_scores))
        result_max_val = n_most_similar_score_result[result_saved_maximum_score_index]

        # -------------------------------------------------
        # draw result
        # -------------------------------------------------
        result_img = 255 * np.ones((20, 770, 3), np.uint8)
        for each_element in n_most_similar_score_result:
            label_image_path = each_element["label_image_path"]
            labeled_img = cv2.imread(label_image_path)  # each image on the label set
            similarity_score_last, diff_last = self.structural_similarity_score_diff(self.blob_roi_image.copy(),
                                                                                     labeled_img.copy())
            image_structural_similarity = self.visualize_structural_similarity(self.blob_roi_image.copy(),
                                                                               labeled_img.copy(),
                                                                               label_image_path, similarity_score_last,
                                                                               diff_last)
            result_img = np.concatenate((result_img, image_structural_similarity), axis=0)
        return n_most_similar_score_result, result_max_val, result_img


labeled_image_folder = "./chagas-capilares/imagenes/gris/si/"
blob_roi_image = cv2.imread('./chagas-capilares/imagenes/gris/si/chagas3_subimagen2152.png')
n_most_similar_score, max_result, result_img = SimilarityLabelClass(blob_roi_image, labeled_image_folder)\
    .bring_n_number_most_similar_label_result()

cv2.imwrite("image_structural_similarity_result.png", result_img)
cv2.imshow('image_structural_similarity_result', result_img)
cv2.waitKey(0)

print("n_most_similar_score_result", n_most_similar_score)
print("max_result", max_result)

# görsel seti benzerliğine göre kümalenip , her kümeden ilk görsel değerlendirildikten sonra küme
# üzerinden bu benzerlik işlemi yapılsa? set geniş ise işlemi hızlandırır

# https://resulsilay.medium.com/opencv-y%C3%BCz-tan%C4%B1ma-structural-similarity-ssim-mse-a0da0eb24e0c
# https://stackoverflow.com/questions/71567315/how-to-get-the-ssim-comparison-score-between-two-images
