import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid


def method1(image):
    original_img = image.copy()
    kernel = np.ones((8, 8), np.uint8)
    image = cv2.erode(image, kernel, iterations=2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 50, 50)

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    i = 0
    for contour in contours:
        contour_formatted = [[cnt[0][0], cnt[0][1]] for cnt in contour]

        if len(contour_formatted) >= 4:
            pts = np.array(contour_formatted, np.int32)
            pts = pts.reshape((-1, 1, 2))
            convex_hull = cv2.convexHull(pts)

            pts = np.squeeze(contour_formatted)
            each_polygon = Polygon(pts)
            # https://stackoverflow.com/questions/20833344/fix-invalid-polygon-in-shapely
            each_polygon = make_valid(each_polygon)

            # if each_polygon.area > 50:
            poly_center = list(each_polygon.centroid.coords[0])
            int_poly_center = (int(poly_center[0]), int(poly_center[1]))
            # cv2.circle(original_img, int_poly_center, 10, (0, 0, 255), -1)

            cv2.drawContours(original_img, [pts], -1, (0, 0, 255), 2)
            # font, fontScale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            # cv2.putText(original_img, str(each_polygon.area), tuple(int_poly_center),
            #             font, fontScale, (255, 0, 0), thickness, cv2.LINE_AA)

    cv2.imwrite('area_segmented_image2.jpg', original_img)
    cv2.imshow('Contours', original_img)
    cv2.waitKey(0)


def method2(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = image.reshape((-1, 3))  # numpy reshape operation -1 unspecified
    # Convert to float type only for supporting cv2.kmean
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.45)
    k = 5  # Choosing number of cluster
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]  # Mapping labels to center points( RGB Value)
    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((image.shape))
    cv2.imshow('segmented_image', segmented_image)
    cv2.imwrite('area_segmented_image.jpg', segmented_image)
    cv2.waitKey(0)



image = cv2.imread("denememem.jpg")
method1(image)
