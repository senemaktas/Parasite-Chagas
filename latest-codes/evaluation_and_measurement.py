import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import keras
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# from matplotlib.backends.backend_agg import FigureCanvasAgg
# from matplotlib.figure import Figure


def create_batch_of_image(rois):
    batch_of_img = []
    for roi in rois:
        img = cv2.resize(roi, (75, 75), interpolation=cv2.INTER_AREA)
        img = img / 255.0
        batch_of_img.append(img)
    batch_of_img = np.asarray(batch_of_img)
    return batch_of_img


def keep_counting_predictions(model, batch_of_roi, prev_pred_val=[]):
    current_val = prev_pred_val   # will update
    for roi in batch_of_roi:
        # roi = cv2.fastNlMeansDenoisingColored(roi, None, 10, 10, 7, 21)
        # roi = cv2.resize(roi, (75, 75), interpolation=cv2.INTER_AREA)
        roi = roi.reshape(1, 75, 75, 3)  # return the image with shaping that TF wants.
        # roi = np.array(roi) / 255.0
        # tf.keras.applications.xception.preprocess_input( x, data_format=None)
        # roi = tf.keras.applications.inception_resnet_v2.preprocess_input(roi)
        # roi = tf.keras.applications.inception_v3.preprocess_input(roi)
        yhat = model.predict(roi)
        print("\nyhat", yhat)
        #  y_class = yhat.argmax(axis=-1)
        y_class = np.argmax(yhat, axis=1)
        current_label = "no" if int(y_class) == 0 else "si"
        print("current_label", current_label)
        current_val.append(current_label)
    # current_val = prev_pred_val   # will update
    # yhat = model.predict_on_batch(batch_of_roi)
    # y_class = yhat.argmax(axis=-1)
    # for y in y_class:
    #     current_label = "no" if int(y) == 0 else "si"
    #     current_val.append(current_label)

    return current_val


# ------------------------------------
#         measurements - graphs
# ------------------------------------
def get_confusion_matrix(total_prediction_val):
    actual = total_prediction_val
    predicted = ["si"] * len(total_prediction_val)

    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
    cm_display.plot()
    # plt.show()
    return "cm_img"

# !!! occlusion, 3d surface, quick movements create missing parts its normal to say "no" chagas
# we only gets dense optical flow detected ones.


def get_video_metrics(total_prediction_val):

    actual = total_prediction_val
    predicted = ["si"] * len(total_prediction_val)

    # pos_label="si" which by default are 1 for positive case and 0 for negative case
    # Accuracy measures how often the model is correct.
    # (True Positive + True Negative) / Total Predictions
    Accuracy = metrics.accuracy_score(actual, predicted)

    # True Positive / (True Positive + False Positive)
    # Precision does not evaluate the correctly predicted negative cases
    Precision = metrics.precision_score(actual, predicted, pos_label="si")

    # True Positive / (True Positive + False Negative)
    # Sensitivity is good at understanding how well the model predicts something is positive
    Sensitivity_recall = metrics.recall_score(actual, predicted, pos_label="si")

    # 2 * ((Precision * Sensitivity) / (Precision + Sensitivity))
    # This score does not take into consideration the True Negative values
    F1_score = metrics.f1_score(actual, predicted, pos_label="si")
    print({"Accuracy": Accuracy, "Precision": Precision,
           "Sensitivity_recall": Sensitivity_recall, "F1_score": F1_score})

    return {"Accuracy": Accuracy, "Precision": Precision, "F1_score": F1_score}


#     video1 = [98.8, 98.8, 98.8, 98.8]
#     video2 = [98.6, 97.8, 97.0, 96.2]
#     video3 = [98.6, 97.8, 97.0, 96.2]
#     video4 = [98.6, 97.8, 97.0, 96.2]
#     video5 = [98.6, 97.8, 97.0, 96.2]
def create_multiple_video_plot_table_metrics(video1, video2, video3, video4, video5):

    plt.style.use(['seaborn'])
    sns.set(palette='colorblind')
    matplotlib.rc("font", family="Times New Roman", size=12)

    labels = ['Accuracy', 'Precision', 'Sensitivity_recall', 'F1_score']
    bar_width = 0.1

    data = [video1, video2, video3, video4, video5]

    colors = sns.color_palette(palette='colorblind')
    columns = ('Accuracy', 'Precision', 'Sensitivity_recall', 'F1_score')

    index = np.arange(len(labels))
    fig = plt.figure(figsize=(12, 9))
    plt.bar(index, video1, bar_width)
    plt.bar(index+bar_width+.02, video2, bar_width)
    plt.bar(index+2*bar_width+.02, video3, bar_width)
    plt.bar(index+3*bar_width+.02, video4, bar_width)
    plt.bar(index+4*bar_width+.02, video5, bar_width)
    plt.table(cellText=data,
              rowLabels=['video1', 'video2', 'video3', 'video4', 'video5'],
              rowColours=colors,
              colLabels=columns,
              loc='bottom',
              bbox=[0, -0.225, 1, 0.2])   # [left, bottom, width, height]

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.ylabel('Percentage (%)')
    plt.xticks([])
    plt.title('Chagas Video Tracking Evaluation Metrics')
    plt.show()


