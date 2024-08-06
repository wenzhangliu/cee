import torch
import torch.nn.functional as f


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

    c_threshold = 0.8 # causal threshold
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
