import os
import json
# # # error :  Initializing libiomp5md.dll, but found libiomp5md.dll already initialized. icinn
# os.environ['KMP_DUPLICATE_LIB_OK']='0'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "0"  # 'True'
# # comment out below line to enable tensorflow logging outputs
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import tensorflow as tf
# # from tensorflow.keras import mixed_precision
# # # Equivalent to the two lines above
# # mixed_precision.set_global_policy('mixed_float16')
#
# #Setting gpu for limit memory
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     #Restrict Tensorflow to only allocate 10gb of memory on the first GPU
#     try:
#         tf.config.experimental.set_virtual_device_configuration(gpus[0],
#        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#        #virtual devices must be set before GPUs have been initialized
#         print(e)
#  os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:12048"
import torch
# device=torch.device("cuda" if torch.cuda.is_available() else "mps"
# if torch.backends.mps.is_available() else "cpu")
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
torch.cuda.empty_cache()
import gc
gc.collect()
# print(torch.cuda.mem_get_info())
# print(torch.cuda.memory_summary(device=None, abbreviated=False))


from preprocess import ChagasPreProcess  # step 1
from get_background_image_using_video import GetBackgroundImg  # step 2
from dense_opt_flow import DenseOpticalFlow  # step 3
# from extract_flow_areas import ExtractFlowArea  # step 4
from fmo_solution import WriteFMOSolutionAsWanted, ReturnFMOSolution
from custom_point_fmo_tracker import CleanResultFMO
from evaluation_and_measurement import *


def main():
    file_path = "./classification_inceptionv3/inceptionv3_result_metrics.json"
    video_folder_path = "./chagas-capilares/videos"

    for video_name in os.listdir(video_folder_path):
        input_video_path = os.path.join(video_folder_path, video_name) #  "./chagas-capilares/videos/chagas71.MOV"
        out_put_folder = './chagas_out/'

        video_name_with_ext = os.path.basename(input_video_path)
        video_name = video_name_with_ext.split('.')[0]
        video_number = video_name[6:]

        frame_num = 1000000   # for preprocess, dense flow and tracking
        max_frame_num = 1000000  # for background estimation
        skip_x_frames = 100

        # ****************************************************************
        # convert grayscale - sharpen - save as avi as grayscale video
        # ****************************************************************
        print('\nPreprocessing Started ...')
        preprocess_out_path = ChagasPreProcess(input_video_path, out_put_folder, max_frame_num).return_preprocess()
        # ****************************************************************
        #     Background Removing - save as avi as grayscale video
        # ****************************************************************
        print('\nBackground Removing Started ...')
        cleaned_out_path, background_img = GetBackgroundImg(input_video_path, preprocess_out_path, skip_x_frames,
                                                            max_frame_num, out_put_folder).removed_back()

        # ****************************************************************
        # preprocess the images and save the video as avi file
        # ****************************************************************
        print('\nDense Flow Started ... ')
        dense_flow_video_path = DenseOpticalFlow(cleaned_out_path).get_dense_flow()
        # ****************************************************************
        # noise reduction - blob detections - tracking
        # ****************************************************************
        print('\nObject ROI extraction and Tracking Started ...')
        # chagas_tracking_result_path = ExtractFlowArea(dense_flow_video_path, preprocess_out_path,
        #                                               out_put_folder, frame_num).result_main()

        dense_optical_flow_video = dense_flow_video_path  # "./chagas_out/chagas20_preprocess_out_cleaned_out_flow_dense.avi"
        # dense_optical_flow_video = "./chagas_out/chagas66_preprocess_out_cleaned_out_flow_dense.avi"
        max_skipped_frame_num = 90  # after each new added value update to 0
        max_shortest_distance = 25
        total_line_result, point_line_img, result_metrics1 = ReturnFMOSolution(input_video_path, out_put_folder,
                                                                              dense_optical_flow_video,
                                                                              max_skipped_frame_num,
                                                                              max_shortest_distance
                                                                              ).return_fmo_solution()

        cv2.imwrite("denememem.jpg", point_line_img)

        # *********
        remove_lines_less_than = 5
        tracking_result = CleanResultFMO(total_line_result, remove_lines_less_than).get_certain_ids_lines()

        input_video = preprocess_out_path  # = "./chagas_out/chagas66_preprocess_out_cleaned_out.avi"
        result_metrics = WriteFMOSolutionAsWanted(input_video_path, out_put_folder,
                                                  input_video, tracking_result).return_fmo_solution()

        write_each_video_metrics = {str(video_number): result_metrics}

        if not os.path.exists(file_path):
            with open(file_path, "w", encoding='utf-8') as file:
                json.dump(write_each_video_metrics, file, ensure_ascii=False, indent=4)
        else:
            with open(file_path, "a", encoding='utf-8') as file:
                file.write(',')
                file.write('\n')
                json.dump(write_each_video_metrics, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()

