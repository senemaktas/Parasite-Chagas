import cv2
import os
import numpy as np


class ExtractFlowArea:

    def __init__(self, img_path, img_name):
        self.img_path = img_path
        self.img_name = img_name
        self.img = cv2.imread(self.img_path)
        self.e_kernel = (4,4)
        self.d_kernel = (8,8)

    def morph_operations(self):
        # can use different kernels with this rather than opening
        erosion_kernel = np.ones(self.e_kernel, np.uint8)   # 4, 8, 12
        dilation_kernel = np.ones(self.d_kernel, np.uint8)

        # The first parameter is the original image,kernel is the matrix with which image is convolved and
        # third parameter is the number of iterations, which will determine how much you want to erode/dilate
        # a given image.
        frame_erode = cv2.erode(self.img, erosion_kernel, iterations=1)
        out_img = cv2.dilate(frame_erode, dilation_kernel, iterations=1)
        # opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return out_img

    def result_main(self):

        dh, dw, _ = self.img .shape

        kernel_flow_frame = self.morph_operations()

        flow_frame1 = cv2.copyMakeBorder(kernel_flow_frame, top=100, bottom=10, left=10, right=10,
                                        borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.putText(flow_frame1, 'Erode : ' + str(self.e_kernel) + ' Dilate : ' + str(self.d_kernel),
                    (30, 70), cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 0, 0), thickness=6, lineType=cv2.LINE_AA)

        # -------------------------------------------------------
        # together_img_left = np.concatenate((flow_frame1, frameee), axis=1)  # img_list[0]
        # together_img = together_img_left
        output_folder = "E:\senem\chagas_project\morphology_sample_tries\outputs"
        new_img_path = output_folder + "/" + self.img_name + "_" + str(self.e_kernel) + "_" + str(self.d_kernel) + ".png"
        cv2.imwrite(new_img_path, flow_frame1)

        return 0

input_folder = "E:\senem\chagas_project\morphology_sample_tries\inputs"
output_folder = "E:\senem\chagas_project\morphology_sample_tries\outputs"

for img_path in os.listdir(input_folder):
    img_name = os.path.basename(img_path).split('.')[0]
    img_full_path = input_folder + "/" + img_path
    print(img_name)
    print(img_full_path)
    ExtractFlowArea(img_full_path, img_name).result_main()