import os
# # error :  Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'True'
# # comment out below line to enable tensorflow logging outputs
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#  os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:12048"
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
import gc
gc.collect()
import cv2
import numpy as np
print(torch.__version__)
from FastSAM.fastsam import FastSAM
# from fastsam_segment_predict import FastSAMSegmentation
torch.cuda.empty_cache()


class FastSAMSegmentation:

    # def __init__(self, original_image, img_name):
    def __init__(self, original_image):
        # self.img_name = img_name
        self.model_path = "./weights/FastSAM-s.pt"
        self.original_image = original_image  # np.array(original_image).astype(np.float16) / 255.0
        self.retina = True
        self.imgsz = 1024  # 640 1024
        self.conf = 0.4
        self.iou = 0.9

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

    def roi_boxes(self):

        self.original_image = self.morph_operations(self.original_image)

        # load model
        model = FastSAM(self.model_path)
        """ you can’t train your entire model in FP16 because some equations don’t support it, 
        but it still speeds up the process quite a bit. First, you have to convert your 
        model to FP16. To do this you have to call the .half() function like this. """
        # model.half()
        fastsam_result = model(self.original_image, device=device, retina_masks=self.retina,
                               imgsz=self.imgsz, conf=self.conf, iou=self.iou)

        initial_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        send_track_points = []
        send_track_bbox = []

        segments_list = []
        if fastsam_result is not None:
            xywh_boxes = fastsam_result[0].boxes.xywh

            for i in range(len(xywh_boxes)):
                bbox = fastsam_result[0].boxes.xyxy[i]  # for deepsort
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = [0 if (a < 0) else int(a) for a in [x1, y1, x2, y2]]

                center_coordinates = int((x2 + x1) / 2), int((y2+y1)/2)
                send_track_points.append(list(center_coordinates))
                # cv2.circle(self.original_image, center_coordinates, radius=5, color=(0, 0, 255), thickness=-1)
                cv2.rectangle(self.original_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                # bbs = tuple(([x, y, w, h], 0.9, 1))
                bbs = [[x1, y1, x2, y2], 1, 1]  # bytetrack
                send_track_bbox.append(bbs)
        # self.original_image = cv2.copyMakeBorder(self.original_image, top=10, bottom=10, left=10, right=10,
        #                                 borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
        # cv2.imwrite("./fastsam_background_comparison/output/fastsam_" + str(self.img_name) + ".png", self.original_image)
        return send_track_bbox, send_track_points, initial_img


# if __name__ == "__main__":
#     img = cv2.imread("./sample_img.png")
#     anything_segmentation_result = FastSAMSegmentation(img).roi_boxes()

# if __name__ == "__main__":
#     input_folder = "./fastsam_background_comparison/input"
#     for img_path in os.listdir(input_folder):
#         img_name = os.path.basename(img_path).split('.')[0]
#         img_full_path = input_folder + "/" + img_path
#         img = cv2.imread(img_full_path)
#         background_segmentation_result = FastSAMSegmentation(img, img_name).roi_boxes()