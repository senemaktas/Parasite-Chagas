import cv2
import math
import string
import random
import numpy as np


class DistanceMeasurePoint:

    def __init__(self, total_points, current_frame_points):
        self.total_points = total_points
        self.current_frame_points = current_frame_points

    def cosine_dist(self):
        pass

    def euclidian_dist(self, p, q):
        p = [3, 3]
        q = [6, 12]
        euc_dist = math.dist(p, q)
        return euc_dist

    def manhattan_dist(self):
        pass

    def return_shortest_dist(self):
        pass


# -------------------------------------------------------------------------
# TODO: id silindikten sonra update edilebilir??
class CleanResultFMO:

    def __init__(self, whole_total_line, remove_lines_less_than):
        self.whole_total_line = whole_total_line
        self.remove_lines_less_than = remove_lines_less_than

    # to reduce the wrong tracking
    def delete_lines_which_is_not_inside_dense_area(self):
        pass

    @staticmethod
    def random_line_color_generator():
        color = np.random.randint(0, 256, size=3, dtype='uint8').tolist()
        return color

    def get_certain_ids_lines(self):
        ids_and_lines = []
        if len(self.whole_total_line) > 0:
            for each_line_info in self.whole_total_line:
                line_name = list(each_line_info)[0]
                each_line_val_dict = each_line_info[line_name][0]

                start_frame_num = int(each_line_info[line_name][1]["start_frame_num"])
                end_frame_num = int(each_line_info[line_name][1]["end_frame_num"])
                random_line_color = self.random_line_color_generator()
                if len(each_line_val_dict) >= self.remove_lines_less_than:
                    ids_and_lines.append({str(line_name): [{"points": list(each_line_val_dict.values())},
                                                           {"start_frame_num": start_frame_num,
                                                            "end_frame_num": end_frame_num,
                                                            "line_color": random_line_color}]})

        return ids_and_lines


class CustomPointPatternTracking:

    def __init__(self, point_line_img, current_frame_num, current_points, total_line,
                 max_skipped_frame_num, max_shortest_distance):
        self.point_line_img = point_line_img
        self.current_frame_num = current_frame_num
        self.current_points = current_points
        self.total_line = total_line
        self.max_skipped_frame_num = max_skipped_frame_num  # after each new added value update to 0
        self.max_shortest_distance = max_shortest_distance

    @staticmethod
    def check_missed_frames_and_fill(current_line_info, point_value_added):
        line_name = list(current_line_info)[0]
        each_line_val_dict = current_line_info[line_name][0]
        last_added_key = list(each_line_val_dict)[len(each_line_val_dict) - 1]
        last_added_val = each_line_val_dict[str(last_added_key)]
        max_skipped_frame_num = current_line_info[line_name][1]['max_skipped_frame_num']

        how_many_parts = max_skipped_frame_num + 2  # 2 points itself
        splitted_lines = np.linspace(last_added_val, point_value_added, how_many_parts)
        points_will_be_edit = splitted_lines[1:-1]  # 2 points itself

        for newly_point in points_will_be_edit:
            newly_point = list(map(int, newly_point))
            random_str = ''.join(random.choices(string.ascii_letters, k=9))
            current_line_info[line_name][0].update({str(random_str): newly_point})
        return current_line_info

    def return_total_line(self):

        # -----------------------------------------------------------------
        #                        DRAW POINTS
        # -----------------------------------------------------------------
        for point in self.current_points:
            x, y = point
            # cv2.circle(self.point_line_img, (int(x), int(y)), radius=2, color=(0, 0, 255), thickness=-1)

        # -----------------------------------------------------------------------------
        # line a kaydedilen her bir pointe ait color encode u da eklenmelidir.
        # line a eklenme sırası sabit tutulmalıdır. en sonuncu değere göre distance ölçümü yapılacaktır.
        # line ın başladığı ve en son eklendiği frame num tutulmalı. Her eklemede güncellenmeli
        # -----------------------------------------------------------------------------
        if len(self.total_line) == 0:
            for i in range(len(self.current_points)):
                random_str = ''.join(random.choices(string.ascii_letters, k=9))
                self.total_line.append({str(i+1): [{str(random_str): self.current_points[i]},
                                                   {"start_frame_num": int(self.current_frame_num),
                                                   "end_frame_num": int(self.current_frame_num),
                                                    "max_skipped_frame_num": 0}]})
        # -----------------------------------------------------------------
        # eğer ilk frame değilse veya değer varsa zaten devam et
        # -----------------------------------------------------------------
        if len(self.total_line) > 0:
            for each_line_info in self.total_line:
                each_line_info_index = self.total_line.index(each_line_info)
                line_name = list(each_line_info)[0]
                each_line_val_dict = each_line_info[line_name][0]
                each_line_val_dict_last_added_key = list(each_line_val_dict)[len(each_line_val_dict) - 1]
                each_line_val_dict_last_added_val = each_line_val_dict[str(each_line_val_dict_last_added_key)]

                # to compare distance get the latest value of each line
                last_added_point = each_line_val_dict_last_added_val

                # ------------------------------------------------------------------------------------
                # max_skipped_frame_num belirlenen değere ulaşmadıysa line uzatma işlemine devam et...
                # ------------------------------------------------------------------------------------
                if each_line_info[line_name][1]['max_skipped_frame_num'] < self.max_skipped_frame_num:

                    # ------------------------------------------------------------------------------------
                    # her bir şimdiki nokta ile listede var olan noktayı kıyasla
                    # en küçük mesafedeki değeri al ve max_shortest_distance küçük eşit mi kontrol et ....
                    # koşul saglanirsa listeye ekle pointi ve frame numarasini güncelle ...
                    # ------------------------------------------------------------------------------------
                    select_min_val = []
                    for current_point in self.current_points:
                        euc_dist = math.dist(last_added_point, current_point)
                        select_min_val.append(euc_dist)

                    if len(select_min_val) != 0:
                        min_value = min(select_min_val)
                        min_val_index = select_min_val.index(min_value)
                        min_line_point = self.current_points[min_val_index]
                        if min_value <= self.max_shortest_distance:
                            cv2.line(self.point_line_img, last_added_point, min_line_point,
                            color=(255, 255, 0), thickness=5)

                            # ----------------------------------------------------------------------
                            # check missed frames and fill them according to "max_skipped_frame_num"
                            # last added val and with the current point value to be added
                            # ----------------------------------------------------------------------
                            if each_line_info[line_name][1]['max_skipped_frame_num'] > 0:
                                filled_each_line_info = self.check_missed_frames_and_fill(each_line_info, min_line_point)
                                # update the value
                                self.total_line[each_line_info_index][str(line_name)] = filled_each_line_info[str(line_name)]

                            # şimdilik listeden değerleri kaldır ve döngü ile line çizmeye çalış
                            # TODO: iki küçük mü değerlendirmesi yapıp - en küçük alınmalı
                            # şuanda sadece bir tanesi yapılıyor... belki sonradan gelecek listedeki nokta daha yakın??

                            # ----------------------------------------------------------------------
                            # start sabit kalirken end guncellenmeli - line a eklenen değer silinmeli
                            # eşleşen line a değer eklenir - end_frame_num guncellenir - current_pointsden değer silinir
                            # yeni bir değer eklendiği için "max_skipped_frame_num" değeri 0 a ayarlanır ...
                            # ----------------------------------------------------------------------

                            random_str = ''.join(random.choices(string.ascii_letters, k=9))
                            self.total_line[each_line_info_index][str(line_name)][0].update({str(random_str): self.current_points[min_val_index]})
                            self.total_line[each_line_info_index][str(line_name)][1]['end_frame_num'] = int(self.current_frame_num)
                            self.total_line[each_line_info_index][str(line_name)][1]['max_skipped_frame_num'] = 0
                            del self.current_points[min_val_index]
                        # eğer yakin deger bulamadıysa da 0 la.
                        else:
                            self.total_line[each_line_info_index][str(line_name)][1]['max_skipped_frame_num'] += 1
                    # eğer yakin deger bulamadıysa da 0 la.
                    if len(select_min_val) == 0:
                        self.total_line[each_line_info_index][str(line_name)][1]['max_skipped_frame_num'] += 1
            # ----------------------------------------------------------------------------------
            # geriye kalan "current_points" değerlerinin her birini yeni bir line imiş gibi ekle
            # ----------------------------------------------------------------------------------
            if len(self.current_points) != 0:
                for crt_point in self.current_points:
                    random_str = ''.join(random.choices(string.ascii_letters, k=9))
                    self.total_line.append({str(len(self.total_line)+1): [{str(random_str): crt_point},
                                                                          {"start_frame_num": int(self.current_frame_num),
                                                                          "end_frame_num": int(self.current_frame_num),
                                                                           "max_skipped_frame_num": 0}]})
        return self.total_line, self.point_line_img

