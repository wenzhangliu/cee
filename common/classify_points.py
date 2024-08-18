import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


#npm-random
#def classify_points(dist_matrix, threshold=1):
#    n = len(dist_matrix)
#    classes = [-1] * n
#    class_idx = 0
#    for i in range(n):
#        if classes[i] != -1:
#            continue
#        classes[i] = class_idx
#        for j in range(i + 1, n):
#            if max(dist_matrix[i][j],dist_matrix[j][i]) < threshold:
#                classes[j] = class_idx
#        class_idx += 1
#    return classes

#def find_redundant_positions(lst):
#    last_positions = {}
#    for i, x in enumerate(lst):
#        last_positions[x] = i

#    minred_actions_list = []
#    for x in set(lst):
#        class_indices = [i for i, val in enumerate(lst) if val == x]
#        random_index = random.choice(class_indices)
#        minred_actions_list.append(random_index)

#    redundant_actions_list = [i for i, x in enumerate(lst) if i not in minred_actions_list]
#    return minred_actions_list, redundant_actions_list




#******************NPM**************************
# def classify_points(dist_matrix, threshold=1):
#     n = len(dist_matrix)
#     classes = [-1] * n
#     class_idx = 0
#     for i in range(n):
#         if classes[i] != -1:
#             continue
#         classes[i] = class_idx
#         for j in range(i + 1, n):
#             if max(dist_matrix[i][j],dist_matrix[j][i]) < threshold:
#                 classes[j] = class_idx
#         class_idx += 1
#     return classes
#
# # origin code
# def find_redundant_positions(lst):
#     last_positions = {}
#     for i, x in enumerate(lst):
#         last_positions[x] = i
#     minred_actions_list = [i for i, x in enumerate(lst) if i in last_positions.values()]
#     redundant_actions_list = [i for i, x in enumerate(lst) if i not in last_positions.values()]
#     return minred_actions_list, redundant_actions_list


# CEE
def find_indices_of_same_elements(data):
    indices_dict = {}
    for idx, elem in enumerate(data):
        if elem not in indices_dict:
            indices_dict[elem] = [idx]
        else:
            indices_dict[elem].append(idx)
    return indices_dict


def classify_points(dist_matrix, threshold=1):
   n = len(dist_matrix)
   classes = [-1] * n
   class_idx = 0
   for i in range(n):
       if classes[i] != -1:
           continue
       classes[i] = class_idx
       for j in range(i + 1, n):
           if max(dist_matrix[i][j],dist_matrix[j][i]) < threshold:
               classes[j] = class_idx
       class_idx += 1
   classes_classification = find_indices_of_same_elements(classes)
   return classes,classes_classification


def find_redundant_positions(lst, classes_classification, n_matrix):

   c_threshold = 0.8  #threshold
   mini_actions_space = []
   new_matrix = n_matrix.copy()
   for value in list(classes_classification.values()):
       if len(value) > 1:
           s_class_dict = {}
           new_dis = []
           for i in value:
               s_class_dict[i] = new_matrix[i]
           s_class_list = f.softmax(torch.tensor(list(s_class_dict.values())),dim=0)
           for s_value in s_class_list:
               if s_value < c_threshold:
                   s_value = -1e10
                   new_dis.append(s_value)
               else:
                   new_dis.append(s_value)
           if all(value_1 < -1e9 for value_1 in new_dis):
               pass
           else:
               new_s_class_list = f.softmax(torch.tensor(new_dis), dim=0)
               num_samples = 1
               samples = torch.multinomial(torch.Tensor(new_s_class_list),num_samples)
               sampled_index = samples.item()
               mi_actions_index = list(s_class_dict.keys())[sampled_index]
               mini_actions_space.append(mi_actions_index)
       else:
           mini_actions_space.append((list(classes_classification.values())[list(classes_classification.values()).index(value)])[0])
   mini_actions_space.sort(reverse=False)
   redundant_actions_space = [i for i in range(len(lst)) if i not in mini_actions_space]
   return mini_actions_space, redundant_actions_space



#------------------------without classification cee-woc ----------------------
def mini_actions_c(N_matrix):
    cee_threshold = 0.03
    minred_actions_list = []
    redundant_actions_list = []
    new_matrix = N_matrix.copy()
    soft_new = torch.softmax(torch.tensor(new_matrix), dim=0)
    soft_new_list = soft_new.tolist()
    for index, value in enumerate(soft_new_list):
        if value > cee_threshold:
            minred_actions_list.append(index)
        else:
            redundant_actions_list.append(index)

    return minred_actions_list, redundant_actions_list


if __name__ == '__main__':
    kl_path = "D:\\research\\ar_robotics_remotelog\\0317测试新mask\\Mask_unlockpickupar-v0_Nill_PPO_n-1_2023-03-16-17-18-18\\kl_keydoor.txt"
    test_kl = np.loadtxt(kl_path)
    # print(test_kl)
    classes = classify_points(test_kl)
    print(classes)
    minred_actions_list, redundant_actions_list = find_redundant_positions(classes)
    print(minred_actions_list)
    print(redundant_actions_list)
