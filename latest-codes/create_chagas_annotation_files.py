import os
import cv2
import numpy as np
from fastsam_segment_predict import FastSAMSegmentation


def draw_segmentation_result(image, anything_seg_result, color):
    for anything_obj_id in range(len(anything_seg_result)):
        anything_xywh_box = anything_seg_result[anything_obj_id]["xywh_box"]
        anything_confidence_score = anything_seg_result[anything_obj_id]["confidence_score"]
        anything_segmentation_points = anything_seg_result[anything_obj_id]["segmentation_points"]

        # ---------------------------------
        ret, mask_ann_img_binary = cv2.threshold(image, 5, 255, cv2.THRESH_BINARY)
        skin_mask_binary = np.array(anything_segmentation_points)
        inverse_skin_mask_binary = cv2.bitwise_not(skin_mask_binary)
        rgb_inverse_skin_mask_binary = cv2.cvtColor(inverse_skin_mask_binary, cv2.COLOR_GRAY2RGB)
        original_frame_np = np.array(mask_ann_img_binary)
        skin_mask_area_img = cv2.add(original_frame_np, rgb_inverse_skin_mask_binary)
        cv2.imshow("skin_mask_area_img", skin_mask_area_img)
        cv2.waitKey(0)
        # ----------------------------------

        # visualization
        pts = np.array(anything_segmentation_points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=4)
    return image


if __name__ == "__main__":

    dir_img = "../BreastCancerSemanticSegmentationDB/masks/"
    for filee in os.listdir(dir_img):
        if filee.endswith(".jpg") or filee.endswith(".png"):
            print("name: ", filee)
            basename_without_ext = os.path.splitext(os.path.basename(filee))[0]
            mask_ann_img = cv2.imread(dir_img + filee)
            img_h, img_w, img_c = mask_ann_img.shape

            # ----------------------------------------------------------
            #                      mask binarize
            # ----------------------------------------------------------
            img = cv2.cvtColor(mask_ann_img, cv2.COLOR_BGR2GRAY)
            # techniques on the input image all pixels value above 120 will be set to 255
            ret, mask_ann_img_binary = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)

            cv2.imwrite("../BreastCancerSemanticSegmentationDB/binarized_masks/"
                        + str(basename_without_ext) + "_binarized.jpg", mask_ann_img_binary)

            # ----------------------------------------------------------
            #    get segmentation polygons with fastsam - save txt
            # ----------------------------------------------------------
            rgb_binary_img = cv2.cvtColor(mask_ann_img_binary, cv2.COLOR_GRAY2RGB)
            anything_segmentation_result = FastSAMSegmentation(rgb_binary_img).get_fastsam_segment_results()
            if anything_segmentation_result is not None:
                rgb_binary_img = draw_segmentation_result(rgb_binary_img, anything_segmentation_result, (0, 255, 0))
            cv2.imshow("ddd", rgb_binary_img)
            cv2.waitKey(0)

            img_segmentation_lines = []
            if anything_segmentation_result is not None:
                for seg_id in range(len(anything_segmentation_result)):
                    anything_segmentation_points = anything_segmentation_result[seg_id]["segmentation_points"]
                    normalized_points = (np.array(anything_segmentation_points).reshape(-1, 2) /
                                         np.array([img_w, img_h])).reshape(-1).tolist()
                    listToStr = ' '.join([str(elem) for elem in normalized_points])
                    each_line = str(1) + " " + str(listToStr) + "\n"
                    img_segmentation_lines.append(each_line)

                with open("../BreastCancerSemanticSegmentationDB/masks_txts/" +
                          str(basename_without_ext) + '.txt', 'w') as img_file:
                    img_file.writelines(img_segmentation_lines)


