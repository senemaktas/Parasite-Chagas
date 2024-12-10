import random
import string


class MeasureChagasLife:

    def __init__(self, prev_ids_and_lines, curr_ids_list, current_frame_num, is_last_frame=False):
        self.prev_ids_and_lines = prev_ids_and_lines
        self.curr_ids_list = curr_ids_list
        self.current_frame_num = current_frame_num
        self.is_last_frame = is_last_frame

    def id_line_update_main(self):
        # if first ids and values save directly -------------------
        if len(self.prev_ids_and_lines) == 0:
            for current_id in self.curr_ids_list:
                random_str = ''.join(random.choices(string.ascii_letters, k=9))
                self.prev_ids_and_lines.append({str(current_id): [{str(random_str): 33333},
                                               {"start_frame_num": int(self.current_frame_num),
                                                "end_frame_num": int(self.current_frame_num),
                                                "max_skipped_frame_num": 0}]})

        # -----------------------------------------------------------------
        # eğer ilk frame değilse veya değer varsa zaten devam et
        # -----------------------------------------------------------------
        current_ids_which_is_inside_prev = []
        for each_curr_id in self.curr_ids_list:
            if len(self.prev_ids_and_lines) > 0:
                for each_line_info in self.prev_ids_and_lines:
                    each_line_info_index = self.prev_ids_and_lines.index(each_line_info)
                    each_line_name = list(each_line_info.keys())[0]
                    # for each_line_name, each_line_val in self.prev_ids_and_lines.items():
                    if int(each_line_name) == each_curr_id:
                        self.prev_ids_and_lines[each_line_info_index][str(each_line_name)][1]['end_frame_num'] \
                            = int(self.current_frame_num)
                        current_ids_which_is_inside_prev.append(int(each_line_name))

        # eger yeni id iseler direkt kaydet
        for each_curr_id in self.curr_ids_list:
            if each_curr_id not in current_ids_which_is_inside_prev:
                random_str = ''.join(random.choices(string.ascii_letters, k=9))
                self.prev_ids_and_lines.append({str(each_curr_id): [{str(random_str): 33333},
                                               {"start_frame_num": int(self.current_frame_num),
                                                "end_frame_num": int(self.current_frame_num),
                                                "max_skipped_frame_num": 0}]})

        return self.prev_ids_and_lines
