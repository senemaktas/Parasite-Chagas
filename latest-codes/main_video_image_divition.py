import os
import cv2
import numpy as np
from PIL import Image
# error :  Initializing libiomp5md.dll, but found libiomp5md.dll already initialized. icinn
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# tf.autograph.set_verbosity(3, True) # to see autograph error with details!!!
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Hide GPU from visible devices tf.config.set_visible_devices([], 'GPU:0')
# chagas_classification_model_cpu_sgd_sigmoid_binary_epoch15
chagas_classification_model_path = r'E:\senem\chagas_project\classification_efficient\gpu_efficient_gray_sgd_001_softmax_binary_epoch15'
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
            roi = cv2.resize(np.array(roi), (70, 70), interpolation=cv2.INTER_AREA)
            roi = np.expand_dims(roi, axis=0)
            yhat = chagas_classification_model.predict(roi, verbose=0)
            y_class = yhat.argmax(axis=-1)
            if int(y_class[0]) == 1:  # means there is chagas
                actual_obj_bboxes.append([box[0], box[1], box[2], box[3]])
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 4)
    return img


def main():
    video_folder_path = "./chagas-capilares/videos"
    out_put_folder = './chagas_out/tile_out'

    for video_name in os.listdir(video_folder_path):
        input_video_path = os.path.join(video_folder_path, video_name)
        video_name_with_ext = os.path.basename(input_video_path)
        video_name = video_name_with_ext.split('.')[0]

        print("video_name", video_name)

        vidcap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        video_width, video_height = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH), vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        out_path = str(out_put_folder) + str(video_name) + '_tile_out.avi'
        out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'MPEG'), fps, (int(video_width), int(video_height)))

        frame_num = 0
        while vidcap.isOpened():
            ret, frame = vidcap.read()
            if ret is not True:
                break
            if ret is True:
                frame_num += 1
                print(frame_num)
                tile_size = 70
                whole_img_classification_img = whole_image_classification(frame, tile_size, tile_size)
                out.write(whole_img_classification_img)

        out.release()
        vidcap.release()


if __name__ == "__main__":
    main()


