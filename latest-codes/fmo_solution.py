import os
import json
from box_segment_with_fastsam import FastSAMSegmentation as PredictBoxes  # option 2 (recommended)
from evaluation_and_measurement import *
from custom_point_fmo_tracker import CustomPointPatternTracking, CleanResultFMO
# from box_segment_with_background import BackgroundSegmentation as PredictBoxes  # option 1 (less recommended)
from PIL import Image

# error :  Initializing libiomp5md.dll, but found libiomp5md.dll already initialized. icinn
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "0"  # 'True'
# # comment out below line to enable tensorflow logging outputs
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Hide GPU from visible devices tf.config.set_visible_devices([], 'GPU:0') chagas_classification_model_cpu_sgd_sigmoid_binary_epoch15
chagas_classification_model_path = r'E:\senem\chagas_project\classification_inceptionv3\inceptionv3_trained_model'
chagas_classification_model = tf.keras.models.load_model(chagas_classification_model_path)


def whole_image_classification(np_array_img, width, height):
        # filename, file_extension = os.path.splitext(input)
        # im = Image.open(os.path.join(images_input_path, input))
        im = Image.fromarray(np_array_img)
        img = np_array_img  # cv2.imread(images_input_path + "/" + input)

        imgwidth, imgheight = im.size
        yPieces = imgheight // height
        xPieces = imgwidth // width

        actual_obj_bboxes = []
        for i in range(0, yPieces):
            for j in range(0, xPieces):
                box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), 255, 1)
                roi = im.crop(box)
                roi = cv2.resize(np.array(roi), (75, 75), interpolation=cv2.INTER_AREA)
                roi = np.expand_dims(roi, axis=0)
                roi = np.array(roi) / 255.0
                yhat = chagas_classification_model.predict(roi, verbose=0)
                y_class = yhat.argmax(axis=-1)
                if int(y_class[0]) == 1:  # means there is chagas
                    actual_obj_bboxes.append([box[0], box[1], box[2], box[3]])
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 4)
        return img


class ReturnFMOSolution:
    def __init__(self, input_video_path, out_put_folder, preprocess_out_cleaned_out_flow_dense,
                 max_skipped_frame_num, max_shortest_distance):
        self.input_video_path = input_video_path
        self.out_put_folder = out_put_folder
        self.dense_optical_flow_video = preprocess_out_cleaned_out_flow_dense
        self.max_skipped_frame_num = max_skipped_frame_num  # after each new added value update to 0
        self.max_shortest_distance = max_shortest_distance

    def return_fmo_solution(self):
        original_vidcap = cv2.VideoCapture(self.input_video_path, cv2.CAP_FFMPEG)
        vidcap = cv2.VideoCapture(self.dense_optical_flow_video, cv2.CAP_FFMPEG)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        video_width, video_height = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH), vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_num = 0
        out_vid_name = os.path.basename(self.dense_optical_flow_video)
        out_vid_name = out_vid_name.split('.')[0]
        fmo_out_path = str(self.out_put_folder) + str(out_vid_name) + '_fmo_out.avi'
        out = cv2.VideoWriter(str(fmo_out_path), cv2.VideoWriter_fourcc(*'MPEG'), fps, (int(video_width), int(video_height)))

        point_line_img = np.zeros((int(video_height), int(video_width), 3), dtype=np.uint8)
        point_line_img[:, :] = [255, 255, 255]

        total_line = []
        # prev_pred_val = []
        while vidcap.isOpened():
            ret, frame = vidcap.read()
            # ret2, original_frame = original_vidcap.read()
            if ret is not True:
                break
            if ret is True:
                frame_num += 1

                # -------------------------------------------------------------
                send_track_bbox, send_track_points, frameee = PredictBoxes(frame).roi_boxes()
                total_line, point_line_img = CustomPointPatternTracking(point_line_img=point_line_img,
                                                                        current_frame_num=frame_num,
                                                                        current_points=send_track_points,
                                                                        total_line=total_line,
                                                                        max_skipped_frame_num=self.max_skipped_frame_num,
                                                                        max_shortest_distance=self.max_shortest_distance
                                                                        ).return_total_line()

                json_object = json.dumps(total_line, indent=4)  # Serializing json
                with open('./chagas_out/contour_' + str(out_vid_name) + ".json", "w") as outfile:  # Writing to sample.json
                    outfile.write(json_object)

                out.write(point_line_img)

                # *********** evaluation *************
        #         if len(send_track_bbox) != 0:
        #             rois = []
        #             for each_box_val in send_track_bbox:
        #                 x1, y1, x2, y2 = each_box_val[0]
        #                 x1, y1, x2, y2 = [0 if (x < 0) else x for x in (x1-15, y1-15, x2+15, y2+15)]
        #                 each_roi = original_frame[y1:y2, x1:x2]
        #                 grayscale_roi = cv2.cvtColor(each_roi, cv2.COLOR_BGR2GRAY)
        #                 each_roi = cv2.merge((grayscale_roi, grayscale_roi, grayscale_roi))
        #                 cv2.imwrite("./classification_folder/flow_rois/original_frame_"+str(frame_num)+".jpg", each_roi)
        #                 rois.append(each_roi)
        #
        #             batch_of_img = create_batch_of_image(rois)
        #             whole_prediction_val = keep_counting_predictions(chagas_classification_model, batch_of_img,
        #                                                              prev_pred_val=prev_pred_val)
        #             # get_confusion_matrix(whole_prediction_val)
        #             prev_pred_val = whole_prediction_val  # update until the last frame
        #
        # metrics_dict = get_video_metrics(prev_pred_val)
        # *********** evaluation *************

        # cv2.imwrite("fmo_area_video_45_point_line_img.png", point_line_img)
        metrics_dict = 0
        out.release()
        vidcap.release()
        original_vidcap.release()
        return total_line, point_line_img, metrics_dict


class WriteFMOSolutionAsWanted:
    def __init__(self, input_video_path, out_put_folder, input_video, ids_and_lines):
        self.input_video_path = input_video_path
        self.out_put_folder = out_put_folder
        self.input_video = input_video
        self.ids_and_lines = ids_and_lines

    # ------------------------------------------------------------------
    # listedeki her bir değer, start_frame_num vve end_frame_num
    # kontrol edilerek, uygun indisteki değeri alınır (line_color ile).
    # ------------------------------------------------------------------
    def return_frame_base_ids(self, current_frame_num):
        current_ids_and_points = {}

        if len(self.ids_and_lines) > 0:
            for each_line_info in self.ids_and_lines:
                id_num = list(each_line_info)[0]
                id_points = each_line_info[id_num][0]
                start_frame_num = int(each_line_info[id_num][1]["start_frame_num"])
                end_frame_num = int(each_line_info[id_num][1]["end_frame_num"])
                line_color = each_line_info[id_num][1]["line_color"]

                if end_frame_num >= current_frame_num >= start_frame_num:
                    point_index = current_frame_num - start_frame_num
                    current_frame_point = id_points["points"][point_index]
                    current_ids_and_points[id_num] = [current_frame_point, line_color]
                else:
                    pass
        return current_ids_and_points

    def return_fmo_solution(self):
        original_vidcap = cv2.VideoCapture(self.input_video_path, cv2.CAP_FFMPEG)
        vidcap = cv2.VideoCapture(self.input_video, cv2.CAP_FFMPEG)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        video_width, video_height = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH), vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        current_frame_num = 0
        out_vid_name = os.path.basename(self.input_video)
        out_vid_name = out_vid_name.split('.')[0]
        fmo_out_path = str(self.out_put_folder) + str(out_vid_name) + '_result_out.avi'
        out = cv2.VideoWriter(str(fmo_out_path), cv2.VideoWriter_fourcc(*'MPEG'), fps, (int(video_width), int(video_height)))

        prev_pred_val = []
        while vidcap.isOpened():
            ret, frame = vidcap.read()
            ret2, original_frame = original_vidcap.read()
            if ret is not True:
                break
            if ret is True:
                current_frame_num += 1
                dh, dw, _ = frame.shape
                # -------------------------------------------------------------
                current_ids_and_points = WriteFMOSolutionAsWanted.return_frame_base_ids(self, current_frame_num)

                # # {'2':[[1046, 843], [145, 203, 90]],'5': [[88, 686],[89, 34, 97]], '7':[[1050, 579],[117, 77, 13]]}
                # *********** evaluation *************
                if len(current_ids_and_points) != 0:
                    rois = []
                    for each_point_val in current_ids_and_points.values():
                        x, y = each_point_val[0][:2]
                        x1, y1, x2, y2 = [0 if (x < 0) else x for x in (x - 30, y - 30, x + 30, y + 30)]
                        each_roi = original_frame[y1:y2, x1:x2]
                        # grayscale_roi = cv2.cvtColor(each_roi, cv2.COLOR_BGR2GRAY)
                        # each_roi = cv2.merge((grayscale_roi, grayscale_roi, grayscale_roi))
                        cv2.imwrite("./classification_folder/flow_rois/original_frame_"
                                    + str(current_frame_num) + ".jpg", each_roi)
                        rois.append(each_roi)

                    batch_of_img = create_batch_of_image(rois)
                    whole_prediction_val = keep_counting_predictions(chagas_classification_model,
                                                                     batch_of_img,
                                                                     prev_pred_val=prev_pred_val)
                    # get_confusion_matrix(whole_prediction_val)
                    prev_pred_val = whole_prediction_val  # update until the last frame

                # *****
                tile_size = 70
                # whole_img_classification_img = whole_image_classification(frame, tile_size, tile_size)
                whole_img_classification_img = frame
                for id, point_color in current_ids_and_points.items():
                    point = point_color[0]
                    point_color = point_color[1]
                    cv2.putText(whole_img_classification_img, str(id), tuple(point), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1.5, color=point_color, thickness=3, lineType=cv2.LINE_AA)
                    cv2.circle(whole_img_classification_img, tuple(point), radius=1, color=point_color, thickness=2)
                out.write(whole_img_classification_img)

        metrics_dict = get_video_metrics(prev_pred_val)
        out.release()
        vidcap.release()
        return metrics_dict


# dense_optical_flow_video = "./chagas_out/chagas20_preprocess_out_cleaned_out_flow_dense.avi"
# out_put_folder = "./chagas_out/"
# max_skipped_frame_num = 30  # after each new added value update to 0
# max_shortest_distance = 10
# total_line_result = ReturnFMOSolution(out_put_folder, dense_optical_flow_video,
#                                       max_skipped_frame_num, max_shortest_distance).return_fmo_solution()
#
# remove_lines_less_than = 15
# tracking_result = CleanResultFMO(total_line_result, remove_lines_less_than).get_certain_ids_lines()
#
# input_video = "./chagas_out/chagas20_preprocess_out.avi"
# WriteFMOSolutionAsWanted(out_put_folder, input_video, tracking_result).return_fmo_solution()


