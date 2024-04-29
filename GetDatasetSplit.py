import random
import time
import sys


#五重交叉验证的划分
def dataset_split_cv(pos_data_list, pos_label_list, neg_data_list,neg_label_list,fold,seed):
    """Split dataset for cross validation.
    :param label_list: list of labels.
    :param _data_list: list of vectors.
    :param fold: the fold of cross validation.
    """
    print("seed:" + str(seed))

    pos_length = len(pos_label_list)
    pos_len_part = (pos_length + 1) / fold
    print("pos_length:" + str(pos_length))

    neg_length = len(neg_label_list)
    neg_len_part = (neg_length + 1) / fold
    print("neg_length:" + str(neg_length))

    split_data_list = []
    split_label_list = []

    count = 1
    while (count <= fold - 1):
        pos_part_data_list = []
        pos_part_label_list = []

        for i in range(int(pos_len_part)):
            random.seed(seed)
            index = random.sample(list(range(len(pos_data_list))), 1)
            pos_data_elem = pos_data_list.pop(index[0])
            pos_part_data_list.append(pos_data_elem)

            label_elem = pos_label_list.pop(index[0])
            pos_part_label_list.append(label_elem)

        neg_part_data_list = []
        neg_part_label_list = []

        for i in range(int(neg_len_part)):
            random.seed(seed)
            index = random.sample(list(range(len(neg_data_list))), 1)
            neg_data_elem = neg_data_list.pop(index[0])
            neg_part_data_list.append(neg_data_elem)

            label_elem = neg_label_list.pop(index[0])
            neg_part_label_list.append(label_elem)



        part_data_list = pos_part_data_list + neg_part_data_list
        part_label_list = pos_part_label_list + neg_part_label_list
        print(str(count) + '\t' + "pos_part_data_list:" + str(len(pos_part_data_list)) + '\t' + "neg_part_data_list:" + str(len(neg_part_data_list)) + '\t' + "part_data_list:" + str(len(part_data_list)))
        if(len(part_data_list)!=len(part_label_list) or len(pos_part_data_list)!=len(pos_part_label_list) or len(neg_part_data_list)!=len(neg_part_label_list)):
            print('lens are not match')
            exit()

        split_data_list.append(part_data_list)
        split_label_list.append(part_label_list)

        count += 1

    data_list = pos_data_list + neg_data_list
    label_list = pos_label_list + neg_label_list
    print(str(count) + '\t' + "pos_part_data_list:" + str(len(pos_data_list)) + '\t' + "neg_part_data_list:" + str(
        len(neg_data_list)) + '\t'  + "data_list:"+ str(len(data_list)))


    if len(data_list) != 0:
        split_data_list.append(data_list)
        split_label_list.append(label_list)

    sys.stdout.flush()

    return split_data_list, split_label_list

