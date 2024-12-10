import cv2
import os
import json
import numpy as np
from pathlib import Path

import torch

# from deep_sort_realtime.deepsort_tracker import DeepSort
from box_segment_with_fastsam import FastSAMSegmentation as PredictBoxes  # option 2 (recommended)
# from box_segment_with_background import BackgroundSegmentation as PredictBoxes  # option 1 (less recommended)
# from ROI_extraction import SkimageBlobDetections, BackgroundSegmentation   # old
from deepsort_stronsort_chagas_life_time_calc import MeasureChagasLife
from evaluation_and_measurement import *

# from strongsort import StrongSORT  # https://github.com/kadirnar/strongsort-pip/issues/5
from strongsort.strong_sort import StrongSORT  # from strongsort import StrongSORT
# tracker_blob_log = StrongSORT(model_weights=Path('osnet_x0_25_msmt17.pt'), device='cpu', fp16=False)

# conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
# max_age : Maximum number of missed before a track is deleted.
# embedder="mobilenet" Choice of ['mobilenet', 'torchreid', 'clip_RN50', 'clip_RN101',
# 'clip_RN50x4', 'clip_RN50x16', 'clip_ViT-B/32', 'clip_ViT-B/16']
# https://pypi.org/project/deep-sort-realtime/
# https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
# tracker_blob_log = DeepSort(max_age=30, max_cosine_distance=0.2, embedder="mobilenet", gating_only_position=False)

# from bytetracker import BYTETracker


class ExtractFlowArea:

    def __init__(self, dense_flow_video_path,  preprocess_out_path, out_put_folder, frame_num):
        self.dense_flow_video_path = dense_flow_video_path
        self.preprocess_out_path = preprocess_out_path
        self.out_put_folder = out_put_folder
        self.frame_num = frame_num

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

    @staticmethod
    def draw_track_function(img, tracks_result):
        current_id_list = []
        cv2.putText(img, "number of tracks : " + str(len(tracks_result)), (25, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        # deepsort
        # for track in tracks_result:
        #     print("track", track)
        #     if not track.is_confirmed():
        #         continue
        #     track_id = track.track_id
        #     current_id_list.append(track_id)
        #     ltrb = track.to_ltrb()  # bounding box format `(min x, miny, max x, max y)`
        #     # cv2.rectangle(img, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 3)
        #     cv2.putText(img, str(track_id), (int(ltrb[0]), int(ltrb[1])), cv2.FONT_HERSHEY_SIMPLEX,
        #                 fontScale=2, color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)

        # stronsort [x1, y1, x2, y2, track_id, class_id, conf]
        for track in tracks_result:
            track_id = track[4]
            current_id_list.append(int(track_id))
            ltrb = track[:3]
            # cv2.rectangle(img, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 3)
            cv2.putText(img, str(int(track_id)), (int(ltrb[0]), int(ltrb[1])), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2, color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)

        return img, current_id_list

    @staticmethod
    def stronsort_func1(send_track_bbox):
        bbox_xywh = []
        confidences = []
        classes = []
        xx = []
        for each_element in send_track_bbox:
            bbox_xywh.append(np.array(each_element[0]))
            confidences.append(each_element[1])
            classes.append(each_element[2])
            x1, y1, x2, y2 = each_element[0]
            # xx.append(np.array([x1, y1, x2, y2, each_element[1], each_element[2]], dtype=object))
            xx.append([x1, y1, x2, y2, each_element[1], each_element[2]])
        return xx
        # return np.array(bbox_xywh), np.array(confidences), np.array(classes)

    def result_main(self):
        # tracker_blob_log = DeepSort(max_age=20, max_cosine_distance=0.2, embedder="torchreid", nn_budget=100,
        #                             gating_only_position=False, bgr=False, nms_max_overlap=0.6)

        tracker_blob_log = StrongSORT(model_weights=Path('osnet_x0_25_msmt17.pt'), device='cpu', fp16=False)
        # track_thresh=0.45, track_buffer=25, match_thresh=0.8, frame_rate=30
        # tracker = BYTETracker()

        vidcap2 = cv2.VideoCapture(self.preprocess_out_path, cv2.CAP_FFMPEG)  # preprocessed video
        vidcap1 = cv2.VideoCapture(self.dense_flow_video_path, cv2.CAP_FFMPEG)   # flow dense video
        fps = vidcap2.get(cv2.CAP_PROP_FPS)
        video_width = vidcap2.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        video_height = vidcap2.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        frame_num = 0

        last_frame_num = vidcap1.get(cv2.CAP_PROP_FRAME_COUNT)

        out_vid_name = os.path.basename(self.dense_flow_video_path)
        out_vid_name = out_vid_name.split('.')[0]
        chagas_tracking_result_path = str(self.out_put_folder) + out_vid_name + "_strongsort_roi_tracking.avi"

        out = cv2.VideoWriter(chagas_tracking_result_path, cv2.VideoWriter_fourcc(*'MPEG'),
                              fps, (int(video_width), int(video_height)))

        prev_ids_and_lines = []

        while vidcap2.isOpened():
            ret, flow_frame = vidcap1.read()
            ret2, original_frame = vidcap2.read()
            if ret is not True:
                break
            if ret is True:
                frame_num += 1
                print("frame_num", frame_num)
                dh, dw, _ = flow_frame.shape

                flow_frame1 = original_frame

                # ------------------------------------------------------
                # create roi from point to track
                # ------------------------------------------------------
                send_track_bbox, send_track_points, frameee = PredictBoxes(flow_frame).roi_boxes()  # new
                # send_track_bbox, frameee = BackgroundSegmentation(flow_frame).back_segment()  # old
                # dets = [[[3, 2, 2 + 2, 2 + 2], 0.9, 1], [[3, 2, 2 + 2, 2 + 2], 0.9, 1], [[3, 2, 2 + 2, 2 + 2], 0.9, 1]]
                # xyxys = dets[:, 4]
                # Update the tracker with the detections
                #  A polygon defined as a ndarray-like [x1,y1,x2,y2,...].
                # tracks = tracker_blob_log.update_tracks(send_track_bbox, frame=frameee)  # deepsort

                if len(send_track_bbox) != 0:
                    det = self.stronsort_func1(send_track_bbox)
                    det = np.array(det)
                    # strongsort update(self, bbox_xywh, confidences, classes, ori_img)
                    tracks = tracker_blob_log.update(det, frameee)  # stronsort
                    # for i, det in enumerate(send_track_bbox):
                    #     det[i] = tracker_blob_log[i].update(send_track_bbox, frameee)
                    # tracks = tracker_blob_log.update(send_track_bbox, _)
                    # frameee = self.draw_track_function(frameee, tracks)  # deepsort
                    frameee, curr_id_list = self.draw_track_function(flow_frame1, tracks)  # deepsort
                    # frameee, curr_id_list = self.draw_track_function(flow_frame1, tracks)  # deepsort
                    # frameee = cv2.copyMakeBorder(frameee, top=100, bottom=10, left=10, right=10,
                    #                              borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    # ---------------------------------------------------------------------------
                    current_frame_num = frame_num
                    is_last_frame = False
                    if current_frame_num == int(last_frame_num):
                        is_last_frame = True
                    updated_prev_ids_and_lines = MeasureChagasLife(prev_ids_and_lines, curr_id_list,
                                                                   current_frame_num,
                                                                   is_last_frame=is_last_frame).id_line_update_main()
                    prev_ids_and_lines = updated_prev_ids_and_lines
                    # -------------------------------------------------------
                    if len(send_track_bbox) != 0:
                        for track in send_track_bbox:
                            boxx = track[0]
                            cv2.rectangle(frameee, (int(boxx[0]) - 20, int(boxx[1]) - 20), (int(boxx[2]), int(boxx[3])), (155, 255, 0), 3)

                    # if frame is last frame then save the values
                    if is_last_frame is True:
                        json_object = json.dumps(prev_ids_and_lines, indent=4)  # Serializing json
                        with open('./dense_opt_out/strongsort_fastsam_out/' + str(out_vid_name) + ".json","w",
                                  encoding="utf8") as outfile:  # Writing to sample.json
                            outfile.write(json_object)

                out.write(frameee)
                if frame_num == self.frame_num:
                    break

        out.release()
        vidcap1.release()
        vidcap2.release()
        return chagas_tracking_result_path


def main():
    video_folder_path = "./dense_opt_out/videos"
    real_videos_path = "./chagas-capilares/videos/"
    for video_name in os.listdir(video_folder_path):
        input_video_path = os.path.join(video_folder_path, video_name) #  "./chagas-capilares/videos/chagas71.MOV"
        dense_flow_video_path = input_video_path  # "./dense_opt_out/chagas20_preprocess_out_cleaned_out_flow_dense.avi"

        real_match_video_name = video_name.split('_')[0].lower()
        real_vid_path = real_videos_path + str(real_match_video_name) + ".MOV"
        print(real_vid_path)
        out_put_folder = "./dense_opt_out/strongsort_fastsam_out/"
        ExtractFlowArea(dense_flow_video_path,  real_vid_path, out_put_folder, frame_num=300000).result_main()


if __name__ == "__main__":
    main()