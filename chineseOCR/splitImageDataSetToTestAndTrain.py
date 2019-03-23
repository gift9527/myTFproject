import os
import random
import shutil

test_dir = "/home/taoming/data/chineseOCR/testSample"
train_dir = "/home/taoming/data/chineseOCR/trainSample"
origin_dir = "/home/taoming/data/chineseOCR/sample"



def get_all_dirs(dir_path):
    dir_list = []
    for root, dirs, files in os.walk(dir_path):
        dir_list.append(root)
    dir_list = dir_list[1:]
    return dir_list


def get_all_files(dir_path):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        file_list = files + file_list
    return file_list


def split_list_random(raw_list, split_number):
    a = raw_list
    b = random.sample(a, split_number)
    c = list(set(a) - set(b))
    return b, c


def copy_file(origin_path, target_path):
    target_dir = "/".join(target_path.split("/")[:-1])
    print (target_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    shutil.copy(origin_path,target_path)





if __name__ == "__main__":
    print (1)
    dirs_list = get_all_dirs(origin_dir)
    dirs_iter = iter(dirs_list)
    try:
        while True:
            dir = next(dirs_iter)
            files_list = get_all_files(dir)
            test_file_list, train_file_list = split_list_random(files_list, 10)
            test_file_iter = iter(test_file_list)
            train_file_iter = iter(train_file_list)
            files_iter = iter(files_list)
            try:
                while True:
                    test_file = next(test_file_iter)
                    origin_file_path = os.path.join(dir, test_file)
                    origin_dir_list = origin_file_path.split('/')
                    last_dir = origin_dir_list[-2]
                    last_file = origin_dir_list[-1]
                    target_file_path = os.path.join(test_dir,last_dir)
                    target_file_path = os.path.join(target_file_path,last_file)
                    copy_file(origin_file_path,target_file_path)
            except StopIteration:
                pass

            try:
                while True:
                    train_file = next(train_file_iter)
                    origin_file_path = os.path.join(dir, train_file)
                    origin_dir_list = origin_file_path.split('/')
                    last_dir = origin_dir_list[-2]
                    last_file = origin_dir_list[-1]
                    target_file_path = os.path.join(train_dir,last_dir)
                    target_file_path = os.path.join(target_file_path,last_file)
                    copy_file(origin_file_path,target_file_path)
            except StopIteration:
                pass
    except StopIteration:
        print ('dirs_iteration Stop')

        # test_list, train_list = split_list_random(files, 10)
        # print test_list
        # print train_list
