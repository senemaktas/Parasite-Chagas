import numpy as np
import itertools

class CalculationResultShow():

    pass

def get_consecutive_missing_values(dictionary, N):
    # # dictteki ilk list elemanla yeni bir liste oluştur.
    # first_element_key = next(iter(updated_dict))
    # first_element_key_val = dictionary[str(first_element_key)]
    # first_elements_dict = dict((x, first_element_key_val.count(x)) for x in set(first_element_key_val))

    missing_values = {}
    for key, value in dictionary.items():
        for val in value:
            if missing_values.get(val) is not None:
                # eğer ilk dict te aynı element varsa değerini 1 arttır
                # burada if li yaı olmalı
                missing_values[val].extend([False])
            if missing_values.get(val) is None:  # ilk değer
                # eğer yer almıyorsa 1 değeri ile ekle
                missing_values[val] = [True]

    # sıralı bir şekilde N sayısı kadar false varsa kayıp

    print("dictionary", dictionary)
    print("missing_values", missing_values)
    return missing_values


prev_frame_ids1 = [1, 2, 3, 4, 5, 6, 7]
prev_frame_ids2 = [1, 2, 4, 6, 7]
prev_frame_ids3 = [1, 2, 5, 6, 7]
prev_frame_ids4 = [1, 2, 3, 5, 6]
prev_frame_ids5 = [1, 2, 5]
current_frame_ids = [2, 3, 5, 7, 8, 9]

frame_ids = [[1, 2, 3, 4, 5, 6, 7], [1, 2, 4, 6, 7], [1, 2, 5, 6, 7], [1, 2, 3, 5, 6], [1, 2, 5]]

# ------------------------------------------------------------
# if id lost for N numbers frames (included current frame) - chagas died does not exist
# ------------------------------------------------------------
keep_max_num_frame_ids = 5
N_num_frames = 3  # not: keep_max_num_frame_ids >= N_num_frames
current_frame_num = 0

id_dict = {}
if len(id_dict.keys()) < keep_max_num_frame_ids:
    for each_frame_index in range(len(frame_ids)):
        id_dict[str(each_frame_index)] = frame_ids[each_frame_index]  # fill with values

updated_dict = id_dict.copy()
if len(updated_dict.keys()) == keep_max_num_frame_ids:
    print("Equal to ", keep_max_num_frame_ids, " element. Oldest frame values are deleted ..")
    # first value key is the oldest frame - delete
    first_key = next(iter(updated_dict))
    oldest_removed_value = updated_dict.pop(str(first_key), None)
    # insert current frame values as last element
    updated_dict[str(keep_max_num_frame_ids)] = current_frame_ids

# how many chages died
if len(updated_dict.keys()) >= keep_max_num_frame_ids:
    no_exists_chagas = get_consecutive_missing_values(updated_dict, N_num_frames)

# ------------------------------------------------------------
# current living chagas number - kaybolanlar dahil mi?
# ------------------------------------------------------------
current_existing_chagas_num = len(current_frame_ids)

# -----------------------------------------------------------------
# new ids appeared - Finding unique values in dictionary values
# -----------------------------------------------------------------
id_dict_val_list = [val for key, val in id_dict.items()]
id_dict_val_list = list(itertools.chain.from_iterable(id_dict_val_list))
previous_frames_unique_ids = np.unique(id_dict_val_list)

# eger current_frame_ids icindeki deger/ degerler get_unique_ids_from_dict içinde yoksa tekil değerleri al
get_current_new_unique_ids = [each_current_id for each_current_id in current_frame_ids
                              if each_current_id not in previous_frames_unique_ids]

print("\nid_dict: ", id_dict)
print("updated_dict: ", updated_dict)
print("previous_frames_unique_ids", previous_frames_unique_ids)  # [28, 30]
print("current_frame_ids: ", current_frame_ids)
print("get_current_new_unique_ids: ", get_current_new_unique_ids)


