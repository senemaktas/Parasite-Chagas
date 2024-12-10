import os
import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
from absl import app  # pip install absl-py
from fastsam_segment_predict import FastSAMSegmentation


class CompleteFASTSAMSegmentations:

    def __init__(self, img, prev_frame_fastsam_result, current_frame_fastsam_result):
        self.img = img
        self.height, self.width, channel = img.shape
        self.prev_frame_fastsam_result = prev_frame_fastsam_result
        self.current_frame_fastsam_result = current_frame_fastsam_result

    def seg_point_to_polygon(self, segmentation_points):
        # x_y_paired_normalized_seg_pnt = list(zip(segmentation_points[::2], segmentation_points[1::2]))
        # each_denormalized_seg_point = [[pnt[0] * self.width, pnt[1] * self.height]
        #                                for pnt in x_y_paired_normalized_seg_pnt]
        # each_denormalized_seg_point = [tuple(a) for a in each_denormalized_seg_point]
        segmentation_polygon = Polygon(segmentation_points)
        # https://stackoverflow.com/questions/20833344/fix-invalid-polygon-in-shapely
        segmentation_polygon = make_valid(segmentation_polygon)
        return segmentation_polygon

    def get_polygon_center_coords(self, polygon_points):
        if len(polygon_points) >= 4:  # en az dort noktasi olmali
            cnt_polygon = self.seg_point_to_polygon(polygon_points)
            poly_center = list(cnt_polygon.centroid.coords[0])
            int_poly_center = (int(poly_center[0]), int(poly_center[1]))
            return int_poly_center

    @staticmethod
    def is_point_in_polygon(current_pol_center, current_frame_polygon, prev_pol_center, prev_frame_polygon):
        # process is done on biggest polygon !!!
        if prev_frame_polygon.area >= current_frame_polygon.area:
            pol1, pol2 = prev_frame_polygon, current_frame_polygon
            pol_center = current_pol_center
        else:
            pol1, pol2 = current_frame_polygon, prev_frame_polygon
            pol_center = prev_pol_center

        x, y = pol_center
        center_point = Point(int(x), int(y))

        if center_point.within(pol1):
            return True
        return False

    def get_extra_segments(self):

        return_extra_segments = []
        for prev_index in range(len(self.prev_frame_fastsam_result)):
            prev_frame_seg_point = self.prev_frame_fastsam_result[prev_index]['segmentation_points']
            prev_frame_seg_center = self.get_polygon_center_coords(prev_frame_seg_point)
            prev_frame_polygon = self.seg_point_to_polygon(prev_frame_seg_point)

            is_prev_match_any = []

            if self.current_frame_fastsam_result is not None:
                for current_index in range(len(self.current_frame_fastsam_result)):
                    current_frame_seg_point = self.current_frame_fastsam_result[current_index]['segmentation_points']
                    current_frame_seg_center = self.get_polygon_center_coords(current_frame_seg_point)
                    current_frame_polygon = self.seg_point_to_polygon(current_frame_seg_point)

                    is_pip = self.is_point_in_polygon(current_frame_seg_center, current_frame_polygon,
                                                      prev_frame_seg_center, prev_frame_polygon)
                    is_prev_match_any.append(is_pip)

                # if all value of is_prev_match_any False then insert segmentation in return_extra_segments
                if not any(is_prev_match_any):
                    return_extra_segments.append(self.prev_frame_fastsam_result[prev_index])
                else:  # eslesen lokasyon varsa es gec suanki prev polygonu
                    pass

            # eger current frame segment yoksa onceki frame segmentlerini direkt buraya tasi
            else:
                return_extra_segments.append(self.prev_frame_fastsam_result[prev_index])
        return return_extra_segments


def draw_segmentation_result(img, anything_segmentation_result, color):
    for anything_obj_id in range(len(anything_segmentation_result)):
        anything_xywh_box = anything_segmentation_result[anything_obj_id]["xywh_box"]
        anything_confidence_score = anything_segmentation_result[anything_obj_id]["confidence_score"]
        anything_segmentation_points = anything_segmentation_result[anything_obj_id]["segmentation_points"]
        # visualization
        pts = np.array(anything_segmentation_points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
    return img


def main(_argv):
    original_video_path = "./chagas_out/chagas20_preprocess_out.avi"
    original_vidcap = cv2.VideoCapture(original_video_path, cv2.CAP_FFMPEG)

    video_path = "./chagas_out/chagas20_preprocess_out_cleaned_out_flow_dense.avi"
    vidcap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    video_width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    video_height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    frame_num = 0

    out_vid_name = os.path.basename(video_path)
    out_vid_name = out_vid_name.split('.')[0]

    out = cv2.VideoWriter("./fastsam_out/" + str(out_vid_name) + 'original_out.avi', cv2.VideoWriter_fourcc(*'DIVX'),
                          fps, (int(video_width), int(video_height)))  # MPEG

    prev_frame_fastsam_result = {}

    while vidcap.isOpened() and original_vidcap.isOpened():
        ret, frame = vidcap.read()
        original_ret, original_frame = original_vidcap.read()
        if (ret or original_ret) is not True:
            break
        if (ret and original_ret) is True:
            dh, dw, _ = frame.shape
            print('Frame #: ', frame_num)

            # --------------------- first frame ---------------------
            if frame_num == 0:
                anything_segmentation_result = FastSAMSegmentation(frame).get_fastsam_segment_results()
                prev_frame_fastsam_result = anything_segmentation_result
                # concatenate pretrained and custom instance detections as only one result
                if anything_segmentation_result is not None:  # to prevent error
                    frame = draw_segmentation_result(original_frame, anything_segmentation_result)
                else:
                    frame = original_frame

            # --------------------- later first frame ---------------------
            if frame_num != 0:
                anything_segmentation_result = FastSAMSegmentation(frame).get_fastsam_segment_results()
                # concatenate pretrained and custom instance detections as only one result
                if anything_segmentation_result is not None:  # to prevent error
                    color1 = (0, 255, 0)
                    frame = draw_segmentation_result(original_frame, anything_segmentation_result, color1)
                else:
                    frame = original_frame

                if prev_frame_fastsam_result is not None:
                    # prev frame segment comparison
                    return_extra_segments = CompleteFASTSAMSegmentations(frame, prev_frame_fastsam_result,
                                                                         anything_segmentation_result).get_extra_segments()
                    if return_extra_segments is not None:
                        color2 = (0, 0, 255)
                        frame = draw_segmentation_result(frame, return_extra_segments, color2)

                prev_frame_fastsam_result = anything_segmentation_result
            frame_num += 1
            out.write(frame)
            if frame_num == 200:
                break

    out.release()
    vidcap.release()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
