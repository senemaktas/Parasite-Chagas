# notes: https://www.geeksforgeeks.org/background-subtraction-in-an-image-using-concept-of-running-average/
# https://cvexplained.wordpress.com/2020/04/17/running-average-model-background-subtraction/

""" The videos under analysis exhibit diverse backgrounds, as illustrated in Fig. \ref{fig:2}. This variability highlights 
the need for tailored parameters to ensure effective analysis for each video. To address the issue, we employ a 
background subtraction technique on each original video. To capture the backgrounds of each video, background 
subtraction running average concept is employed which comprises two main steps: background initialization and 
update. Initially, a model of the background is computed, which is then updated iteratively to adapt to potential 
changes in the scene. In the running average concept, a video sequence is analyzed over a specified set of frames. 
During this sequence, the running average is computed between the current frame and the previous frames, resulting
in a background model. Subsequently, the current frame contains the newly introduced object against the background. 
The absolute difference between the background model, which evolves over time, and the current frame is then computed. 
This difference highlights the newly introduced object, allowing it to be distinguished from the background.In summary,
this algorithm creates a calibrated background image by averaging multiple frames from a video source while giving more 
weight to pixels that remain consistent across frames. Background subtraction - running average concept algorithm is
depicted in the Algorithm 1. """


import numpy as np
import cv2
import os


class GetBackgroundImg:

    def __init__(self, input_video_path, preprocess_out_path, skip_x_frames, max_frame_num, out_put_folder):
        self.background_input_path = input_video_path
        self.preprocess_out_path = preprocess_out_path
        self.out_put_folder = out_put_folder
        self.calibrated_img, self.last_img = None, None
        self.number_of_images, self.count = 0, 0
        self.skip_x_frames = skip_x_frames  # 10
        self.max_frame_num = max_frame_num  # 5000

    def return_background_img(self):
        cap = cv2.VideoCapture(self.background_input_path)
        while cap.isOpened():
            is_frame_received, img = cap.read()

            if is_frame_received and self.count % self.skip_x_frames == 0:
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = img.astype(float)
                self.last_img = img

                if self.calibrated_img is None:
                    self.calibrated_img = img
                else:
                    difference = (self.calibrated_img / self.number_of_images)
                    difference = cv2.absdiff(self.last_img.astype("float"), difference) / 255
                    weights = 1 - difference
                    # weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-21)
                    self.calibrated_img += img * weights

                self.number_of_images += 1
            elif not is_frame_received:
                break

            self.count += 1
            if self.number_of_images > self.max_frame_num:
                break

        calibrated_img = self.calibrated_img / self.number_of_images
        calibrated_img = np.array(calibrated_img, dtype="uint8")
        cap.release()
        return calibrated_img

    def removed_back(self):
        vidcap = cv2.VideoCapture(self.preprocess_out_path, cv2.CAP_FFMPEG)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        video_width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
        video_height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        frame_num = 0

        out_vid_name = os.path.basename(self.preprocess_out_path)
        out_vid_name = out_vid_name.split('.')[0]

        # ****************************************************************
        # convert grayscale - sharpen - save as avi as grayscale video
        # ****************************************************************
        cleaned_out_path = str(self.out_put_folder) + str(out_vid_name) + '_cleaned_out.avi'
        out2 = cv2.VideoWriter(cleaned_out_path, cv2.VideoWriter_fourcc(*'MPEG'),
                               fps, (int(video_width), int(video_height)))  # MPEG DIVX 0 for grayscala
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (image.shape[1], image.shape[0]), 0)

        background_img = self.return_background_img()

        while vidcap.isOpened():
            ret, frame = vidcap.read()
            if ret is not True:
                break
            if ret is True:
                frame_num += 1
                dh, dw, _ = frame.shape
                img_without_background = cv2.absdiff(frame, background_img)
                out2.write(img_without_background)
        out2.release()
        vidcap.release()
        return cleaned_out_path, background_img

# path = "chagas20.MOV"
# background_img = GetBackgroundImg(path=path, skip_x_frames=10, max_frame_num_to_process=5000).return_background_img()
# cv2.imshow("background_img", background_img)
# cv2.waitKey(0)
