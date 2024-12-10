import os
# error :  Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'True'
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
import gc
gc.collect()
from FastSAM.fastsam import FastSAM


class FastSAMSegmentation:

    def __init__(self, original_image):
        self.model_path = "./weights/FastSAM-x.pt"
        self.original_image = original_image
        self.retina = True
        self.imgsz = 1024  # 640 1024
        self.conf = 0.2
        self.iou = 0.99

    def get_fastsam_segment_results(self):

        # load model
        model = FastSAM(self.model_path)
        fastsam_result = model(self.original_image, device=device, retina_masks=self.retina,
                               imgsz=self.imgsz, conf=self.conf, iou=self.iou)

        segments_list = []

        if fastsam_result is not None:
            segments = fastsam_result[0].masks.xy  # consist of unnormalized x,y point pairs
            for each_seg in segments:
                each_denormalized_seg_point = [[int(point[0]), int(point[1])] for point in each_seg]
                segments_list.append(each_denormalized_seg_point)
            #     # visualization
            #     pts = np.array(each_unnormalized_seg_point, np.int32)
            #     pts = pts.reshape((-1, 1, 2))
            #     cv2.polylines(self.original_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            # cv2.imshow("ddd", self.original_image)
            # cv2.waitKey(0)

            xywh_boxes = fastsam_result[0].boxes.xywh
            confidence_scores = fastsam_result[0].boxes.conf
            xywh_boxes = xywh_boxes.int().cpu().numpy().astype('uint8')
            confidence_scores = confidence_scores.float().cpu().numpy().astype('float32')

            anything_segmentation_result = []
            for xywh, confidence_score, segmentation_points in zip(xywh_boxes, confidence_scores, segments_list):
                each_obj = {"xywh_box": xywh, "confidence_score": confidence_score, "segmentation_points": segmentation_points}
                anything_segmentation_result.append(each_obj)

            return anything_segmentation_result
        else:
            return None


# if __name__ == "__main__":
#     img = cv2.imread("./images/3.png")
#     anything_segmentation_result = FastSAMSegmentation(img).get_fastsam_segment_results()
