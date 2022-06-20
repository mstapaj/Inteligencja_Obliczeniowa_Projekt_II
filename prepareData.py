from os import listdir, makedirs
from random import seed, random
from shutil import copyfile
import os
import zipfile

with zipfile.ZipFile("./alzheimer_data.zip", 'r') as zip_ref:
    zip_ref.extractall("./")

dirs = ["./models", "./wrong_predicted", "./divided_data"]

for i in dirs:
    dir = os.path.join(i)
    if not os.path.exists(dir):
        os.mkdir(dir)

src_directory = './data'
res = []

for file in listdir(src_directory):
    if file.split("_")[0].split(".")[0] not in res:
        res.append(file.split("_")[0].split(".")[0])

folder = './data/'
dataset_home = 'divided_data/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
    labeldirs = res
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(newdir, exist_ok=True)

seed(275036)
val_ratio = 0.20

src_directory = './data'
for file in listdir(src_directory):
    src = src_directory + '/' + file
    dst_dir = '/train/'
    if random() < val_ratio:
        dst_dir = 'test/'
    dst = dataset_home + dst_dir + file.split("_")[0].split(".")[0] + '/' + file
    copyfile(src, dst)
