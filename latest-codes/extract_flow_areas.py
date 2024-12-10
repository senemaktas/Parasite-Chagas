import cv2
import os
import numpy as np
from ROI_extraction import SkimageBlobDetections, BackgroundSegmentation
from deep_sort_realtime.deepsort_tracker import DeepSort
# from strongsort import StrongSORT  # from strongsort.strong_sort import StrongSORT
# tracker_blob_log = StrongSORT(model_weights='osnet_x0_25_msmt17.pt', device='cuda', fp16=True)
# conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
# max_age : Maximum number of missed before a track is deleted.
# embedder="mobilenet" Choice of ['mobilenet', 'torchreid', 'clip_RN50', 'clip_RN101',
# 'clip_RN50x4', 'clip_RN50x16', 'clip_ViT-B/32', 'clip_ViT-B/16']
# https://pypi.org/project/deep-sort-realtime/
# https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
# tracker_blob_log = DeepSort(max_age=20, max_cosine_distance=0.2, embedder="mobilenet", gating_only_position=False)

from bytetracker import BYTETracker


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
        cv2.putText(img, "number of tracks : " + str(len(tracks_result)), (25, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        for track in tracks_result:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # bounding box format `(min x, miny, max x, max y)`
            cv2.rectangle(img, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 3)
            cv2.putText(img, str(track_id), (int(ltrb[0]), int(ltrb[1])), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=4, color=(0, 0, 255), thickness=6, lineType=cv2.LINE_AA)
        return img

    def result_main(self):
        tracker_blob_log = DeepSort(max_age=20, max_cosine_distance=0.5, embedder="mobilenet",
                                    gating_only_position=False, bgr=False)

        # track_thresh=0.45, track_buffer=25, match_thresh=0.8, frame_rate=30
        tracker = BYTETracker()

        vidcap2 = cv2.VideoCapture(self.preprocess_out_path, cv2.CAP_FFMPEG)  # preprocessed video
        vidcap1 = cv2.VideoCapture(self.dense_flow_video_path, cv2.CAP_FFMPEG)   # flow dense video
        fps = vidcap2.get(cv2.CAP_PROP_FPS)
        video_width = vidcap2.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        video_height = vidcap2.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        frame_num = 0

        out_vid_name = os.path.basename(self.dense_flow_video_path)
        out_vid_name = out_vid_name.split('.')[0]
        chagas_tracking_result_path = str(self.out_put_folder) + out_vid_name + "_roi_tracking.avi"

        out = cv2.VideoWriter(chagas_tracking_result_path, cv2.VideoWriter_fourcc(*'MPEG'),
                              fps, (int(video_width) + int(video_width)+40, int(video_height)+110))
        # fps, (int(video_width) + int(video_width)+40, int(video_height) + int(video_height)+40))

        while vidcap2.isOpened():
            ret, flow_frame = vidcap1.read()
            ret2, original_frame = vidcap2.read()
            if ret is not True:
                break
            if ret is True:
                frame_num += 1
                dh, dw, _ = flow_frame.shape
                # print('Frame #: ', frame_num)

                # ------------------------------------------------------
                # morphological processes as noise reduction process
                # ------------------------------------------------------
                flow_frame = self.morph_operations(flow_frame)

                # ------------------------------------------------------
                # extend and put name on image - flow dense output image flow_frame
                # ------------------------------------------------------
                flow_frame1 = cv2.copyMakeBorder(original_frame, top=100, bottom=10, left=10, right=10,
                                                borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
                cv2.putText(flow_frame1, 'motiongram flow dense', (25, 45), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

                # ------------------------------------------------------
                # create roi from point to track
                # ------------------------------------------------------
                send_track_bbox, frameee = BackgroundSegmentation(flow_frame).back_segment()
                dets = [[[3, 2, 2 + 2, 2 + 2], 0.9, 1], [[3, 2, 2 + 2, 2 + 2], 0.9, 1], [[3, 2, 2 + 2, 2 + 2], 0.9, 1]]
                xyxys = dets[:, 4]
                tracks = tracker_blob_log.update_tracks(send_track_bbox, frame=frameee)  # deepsort
                tracks = tracker.update(send_track_bbox, _)
                frameee = self.draw_track_function(frameee, tracks)  # deepsort
                frameee = cv2.copyMakeBorder(frameee, top=100, bottom=10, left=10, right=10,
                                             borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
                # -------------------------------------------------------
                together_img_left = np.concatenate((flow_frame1, frameee), axis=1)  # img_list[0]
                together_img = together_img_left
                out.write(together_img)
                if frame_num == self.frame_num:
                    break

        out.release()
        vidcap1.release()
        vidcap2.release()
        return chagas_tracking_result_path

