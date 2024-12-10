import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


# json_folder_path = "E:\senem\chagas_project\chagas_out\\fastsam_out\jsons"
# json_folder_path = "E:\senem\chagas_project\dense_opt_out\deepsort_contour_out\jsons"
json_folder_path = "E:\senem\chagas_project\dense_opt_out\strongsort_fastsam_out\jsons"
whole_static_matrix = []
whole_static_matrix2 = []

for each_file in os.listdir(json_folder_path):
    current_file_path = json_folder_path + "/" + each_file

    list_of_values = []
    with open(current_file_path) as f:
        data = json.load(f)
        for each_tracking in data:
            values_v = list(x for x in each_tracking.values())
            start_end_frame_dif_num = values_v[0][1]["end_frame_num"] - values_v[0][1]["start_frame_num"]
            print(start_end_frame_dif_num)
            tracking_frame_len = len(values_v[0][0])
            if tracking_frame_len >= 0:
                list_of_values.append(start_end_frame_dif_num)

    print("list_of_values", list_of_values)

    mean_list_of_values = np.mean(list_of_values)      # ortalama
    median_list_of_values = np.median(list_of_values)  # ortadaki
    range_list_of_values = np.ptp(list_of_values,axis=0)  # max-min dif

    fps = 20.0
    tracking_second = np.array(list_of_values)/fps

    second_mean_list_of_values = np.mean(tracking_second)      # ortalama
    second_median_list_of_values = np.median(tracking_second)  # ortadaki
    second_range_list_of_values = np.ptp(tracking_second, axis=0)  # max-min dif

    whole_static_matrix.append([len(list_of_values), str(round(mean_list_of_values,3)) + " / " + str(round(second_mean_list_of_values,3)),
                                str(round(median_list_of_values,3)) + " / " + str(round(second_median_list_of_values,3)),
                                str(round(range_list_of_values,3)) + " / " + str(round(second_range_list_of_values, 3))])

    whole_static_matrix2.append([len(list_of_values), mean_list_of_values,
                                median_list_of_values,
                                range_list_of_values])

print(whole_static_matrix)

ylabel_header = ["20", "33", "36", "38", "42", "44", "45", "46", "47", "48", "49", "50",
                 "57", "58", "59", "60", "61", "62", "67", "68", "69", "70", "71", "72"]
xlabel_header = ["Tracked parasite number", "mean","median", "range"]

# df = pd.DataFrame.from_dict(whole_static_matrix, orient='columns', dtype=None)
# df = df.fillna(0)  # nan to zero
from matplotlib.colors import ListedColormap
sns.heatmap(whole_static_matrix2, cmap=ListedColormap(['white']), fmt="", yticklabels=ylabel_header,
            xticklabels=xlabel_header, annot=whole_static_matrix)  # vmin=0, vmax=1, , annot=True

fig, ax = plt.subplots()
plt.yticks()
plt.xticks(rotation=0)
plt.ylabel("Video numbers according to their names")
plt.xlabel("General metrics for total tracked parasites on a video basis")
plt.show()

df = pd.DataFrame(whole_static_matrix)
df.to_excel(excel_writer="./strongsort_fastsam_out.xlsx")

