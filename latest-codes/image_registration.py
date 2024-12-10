import os
import cv2
import numpy as np

# visual odometry python
# ransac RANSAC (Random Sample Consensus)
# https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html

video_path = "  "
out_put_folder = " "

vidcap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
fps = vidcap.get(cv2.CAP_PROP_FPS)
video_width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
video_height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
frame_num = 0

out_vid_name = os.path.basename(video_path)
out_vid_name = out_vid_name.split('.')[0]

# ****************************************************************
#
# ****************************************************************
cleaned_out_path = str(out_put_folder) + str(out_vid_name) + '_cleaned_out.avi'
out2 = cv2.VideoWriter(cleaned_out_path, cv2.VideoWriter_fourcc(*'MPEG'), fps, (int(video_width), int(video_height)))
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (image.shape[1], image.shape[0]), 0)

while vidcap.isOpened():
    ret, frame = vidcap.read()
    if ret is not True:
        break
    if ret is True:
        frame_num += 1
        dh, dw, _ = frame.shape

        # Define a connectivity measure function
        def connectivity_measure(frame1, frame2):
            # Compute optical flow between frame1 and frame2
            flow = cv2.calcOpticalFlowFarneback(frame1, frame2, 0.5, 100)

            # Calculate the average flow magnitude across the ROIs
            rois = ...  # define your ROIs
            avg_flow_magnitude = np.mean([np.sqrt(flow[i].dot(flow[i])) for i in range(len(rois))])

            return avg_flow_magnitude


        # Load a batch of frames
        frames = []
        for i in range(num_frames):
            frame = cv2.imread('frame_{}.jpg'.format(i))
            frames.append(frame)

        # Initialize an empty array to store the connectivity measures
        connectivity = np.zeros((len(frames), len(frames)))

        # Evaluate the connectivity measure between all pairs of frames
        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                connectivity[i][j] = connectivity_measure(frames[i], frames[j])
                connectivity[j][i] = connectivity[i][j]

        # Aggregate the connectivity measures over time
        avg_connectivity = np.mean(connectivity, axis=0)

        print("Average connectivity across batch:", avg_connectivity)


        # # Load the reference and target images
        # reference_image = cv2.imread("reference_image.jpg", cv2.IMREAD_GRAYSCALE)
        # target_image = cv2.imread("target_image.jpg", cv2.IMREAD_GRAYSCALE)
        #
        # # Find keypoints and descriptors using ORB (you can use other methods like SIFT, SURF, etc.)
        # orb = cv2.ORB_create()
        # kp1, des1 = orb.detectAndCompute(reference_image, None)
        # kp2, des2 = orb.detectAndCompute(target_image, None)
        #
        # # Create a brute-force matcher
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # # Match the descriptors
        # matches = bf.match(des1, des2)
        # # Sort the matches by distance
        # matches = sorted(matches, key=lambda x: x.distance)
        # # Keep only the best N matches
        # N = 50
        # matches = matches[:N]
        #
        # # Extract matched keypoints
        # src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        # dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        #
        # # Compute the transformation matrix using RANSAC
        # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        #
        # # Apply the transformation to the target image
        # registered_image = cv2.warpPerspective(target_image, M, (reference_image.shape[1], reference_image.shape[0]))
        #
        # # Save the registered image
        # cv2.imwrite("registered_image.jpg", registered_image)
        #
        # # Display the registered image
        # cv2.imshow("Registered Image", registered_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        out2.write(frame)
out2.release()
vidcap.release()





