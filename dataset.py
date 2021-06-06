import torch
from torchvision import transforms
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(Image.open(self.image_files[index]).convert('RGB')), self.labels[index]


def random_sample(img_filename_list, img_label_list, class_names_list):
    plt.subplots(3, 3, figsize=(8, 8))
    for i, k in enumerate(np.random.randint(len(img_filename_list), size=9)):
        im = Image.open(img_filename_list[k])
        arr = np.array(im)
        plt.subplot(3, 3, i + 1)
        plt.axis('off')
        plt.title(class_names_list[img_label_list[k]])
        plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.savefig('random_sample.pdf')


def load_from_class_folder(data_dir):
    class_names = sorted([x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))])
    num_class = len(class_names)
    image_files = [[os.path.join(data_dir, class_name, x)
                    for x in os.listdir(os.path.join(data_dir, class_name))]
                   for class_name in class_names]
    image_file_list = []
    image_label_list = []
    for i, class_name in enumerate(class_names):
        image_file_list.extend(image_files[i])
        image_label_list.extend([i] * len(image_files[i]))
    num_total = len(image_label_list)
    image_width, image_height = Image.open(image_file_list[0]).size

    print('Total image count:', num_total)
    print("Image dimensions:", image_width, "x", image_height)
    print("Label names:", class_names)
    print("Label counts:", [len(image_files[i]) for i in range(num_class)])
    return num_total, class_names, image_file_list, image_label_list


def load_from_csv(data_dir, ground_truth_file):
    df = pd.read_csv(ground_truth_file)
    class_names = list(df.columns[1:-1])
    num_class = len(class_names)
    image_file_list = [os.path.join(data_dir, img_name)+'.jpg' for img_name in df['image']]
    image_label_list = [list(x[1:]).index(1.0) for _, x in df.iterrows()]
    num_total = len(image_label_list)
    image_width, image_height = Image.open(image_file_list[0]).size

    print('Total image count:', num_total)
    print("Image dimensions:", image_width, "x", image_height)
    print("Label names:", class_names)
    print("Label counts:", [image_label_list.count(i) for i in range(num_class)])
    return num_total, class_names, image_file_list, image_label_list


def load_datasets(img_filename_list, img_label_list, batch_size=64):
    # Split Dataset
    valid_frac, test_frac = 0.1, 0.1
    _datasets = {'train': {'x': [], 'y': []},
                 'val': {'x': [], 'y': []},
                 'test': {'x': [], 'y': []}}

    for i in range(len(img_filename_list)):
        rand_n = np.random.random()
        if rand_n < valid_frac:
            _datasets['val']['x'].append(img_filename_list[i])
            _datasets['val']['y'].append(img_label_list[i])
        elif rand_n < test_frac + valid_frac:
            _datasets['test']['x'].append(img_filename_list[i])
            _datasets['test']['y'].append(img_label_list[i])
        else:
            _datasets['train']['x'].append(img_filename_list[i])
            _datasets['train']['y'].append(img_label_list[i])

    print("Training count =", len(_datasets['train']['x']),
          "Validation count =", len(_datasets['val']['x']),
          "Test count =", len(_datasets['test']['x']))

    # Load to loader
    data_transforms = {
        'train':
            transforms.Compose([
                transforms.RandomResizedCrop(size=64),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
            ]),
        'val':
            transforms.Compose([
                transforms.Resize(size=64),
                transforms.CenterCrop(size=64),
                transforms.ToTensor(),
            ]),
        'test':
            transforms.Compose([
                transforms.Resize(size=64),
                transforms.CenterCrop(size=64),
                transforms.ToTensor(),
            ]),
    }

    _image_datasets = {x: MyDataset(_datasets[x]['x'], _datasets[x]['y'], data_transforms[x]) for x in
                       ['train', 'val', 'test']}
    _data_loaders = {
        x: torch.utils.data.DataLoader(_image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=32) for x in
        ['train', 'val', 'test']}

    _dataset_sizes = {x: len(_image_datasets[x]) for x in ['train', 'val', 'test']}

    return _data_loaders, _dataset_sizes
