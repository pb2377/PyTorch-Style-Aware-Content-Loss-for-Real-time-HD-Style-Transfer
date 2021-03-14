import os
import glob
from torch.utils import data
from torchvision import transforms
from PIL import Image
from random import shuffle
import json


class PlacesDataset(data.Dataset):
    def __init__(self, train=True, input_size=768, style_dataset=None):
        super(PlacesDataset, self).__init__()
        """Initialisation"""
        dataset_dir = '../Datasets/Places365/'
        # if input_size > 256:
        if True:
            data_dir = 'data_large'
            self.min_size = 800.
            self.max_size = 1800.
        # else:
        #     data_dir = 'data_256'
        #     self.min_size = 300.
        #     self.max_size = 1000.

            # print('Cannot Use Original Images, upscaling 256x256 to 768x768')
            # raise NotImplementedError('Only implemented 256x256 dataset')
        self.train = train
        if self.train:
            assert style_dataset is not None
        self.style_dataset = style_dataset
        self.image_dirs = os.path.join(dataset_dir, data_dir)
        self.list_ids = self.get_list_ids()

        if train:
            self.transf = transforms.Compose([
                                              transforms.RandomAffine(degrees=15, shear=0.05, scale=(0.9, 1.1)),
                                              transforms.RandomCrop(input_size),
                                              # transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.2)), # ratio=(1., 1.)),
                                              transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                                              transforms.RandomHorizontalFlip(), #transforms.RandomVerticalFlip(),
                                              transforms.ToTensor(),
                                              ])
        else:
            self.transf = transforms.Compose([# transforms.Resize(input_size),
                                              # transforms.CenterCrop(input_size),
                                              transforms.ToTensor(),
                                              ])

    def get_list_ids(self):
        list_ids = glob.glob(os.path.join(self.image_dirs, '*', '*', '*'))
        list_ids = [i for i in list_ids if '.jpg' in i or '.png' in i]
        return list_ids

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, idx):
        style_image = self.resize_im(self.style_dataset.iterate())
        image = self.resize_im(Image.open(self.list_ids[idx]).convert('RGB'))
        # image = Image.open(self.list_ids[idx]).convert('RGB')
        style_image = self.normalize(self.transf(style_image))
        image = self.normalize(self.transf(image))
        return image, style_image

    @staticmethod
    def normalize(image):
        image *= 2.
        image -= 1.
        return image

    def resize_im(self, image):
        if max(image.selfize) > self.max_size:
            sc = self.max_size / max(image.size)
            image = image.resize((int(sc * image.size[0]), int(sc * image.size[1])), Image.BILINEAR)

        if min(image.size) < self.min_size:
            # Resize the smallest side of the image to 800px
            sc = self.min_size / float(min(image.size))
            if sc < 4.:
                image = image.resize((int(sc * image.size[0]), int(sc * image.size[1])), Image.BILINEAR)
            else:
                image = image.resize((int(self.min_size), int(self.min_size)), Image.BILINEAR)
        return image


class StyleDataset(object):
    def __init__(self, data_dir=None):
        # super(StyleDataset, self).__init__()
        assert data_dir is not None
        # print(data_dir)
        assert os.path.exists(data_dir)
        self.data_dir = data_dir
        self.list_ids = None
        self.iter_ids = None
        self.get_list_ids()
        self.shuffle_ids()

    def get_list_ids(self):
        list_ids = glob.glob(os.path.join(self.data_dir, '*'))
        list_ids = [i for i in list_ids if '.jpg' in i or '.png' in i]
        self.list_ids = list_ids

    def shuffle_ids(self):
        self.iter_ids = [i for i in range(len(self.list_ids))]
        shuffle(self.iter_ids)

    def iterate(self):
        if len(self.iter_ids) < 1:
            self.shuffle_ids()
        idx = self.iter_ids.pop(0)
        image = Image.open(self.list_ids[idx]).convert('RGB')
        return image


class TestDataset(data.Dataset):
    def __init__(self, image_dir='../Datasets/WikiArt-Sorted/data/sample_photographs', input_size=768):
        super(TestDataset, self).__init__()
        self.image_dir = image_dir
        assert os.path.join(self.image_dir)
        self.list_ids = None
        self.get_list_ids()
        self.transf = transforms.Compose([transforms.Resize(int(input_size * 2)),
                                          # transforms.CenterCrop(input_size),
                                          transforms.ToTensor(),
                                          ])

    def get_list_ids(self):
        list_ids = glob.glob(os.path.join(self.image_dir, '*'))
        list_ids = sorted([i for i in list_ids if '.jpg' in i or '.png' in i])
        self.list_ids = list_ids

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, idx):
        image = Image.open(self.list_ids[idx]).convert('RGB')
        image = self.normalize(self.transf(image))
        return image

    @staticmethod
    def normalize(image):
        image *= 2.
        image -= 1.
        return image


class MpiiDataset(data.Dataset):
    """Get dataset for MPII Person DatasetTorch"""

    def __init__(self, train=False, style_dataset=None, input_size=768):
        """Initialisation"""
        dataset_dir = '../Datasets/MPII-Human-Pose/data'
        self.image_dir = os.path.join(dataset_dir, 'images')
        self.train = train
        if self.train:
            assert style_dataset is not None
        self.style_dataset = style_dataset
        self.train = train
        self.input_size = input_size

        jsonfile = os.path.join(dataset_dir, 'mpii_annotations.json')
        with open(jsonfile) as anno_file:
            self.annos = json.load(anno_file)

        self.list_IDs = []
        for idx, val in enumerate(self.annos):
            if os.path.exists(os.path.join(self.image_dir, self.annos[idx]['img_paths'])):
                if train and not val['isValidation']:
                    self.list_IDs.append(idx)
                elif not train and val['isValidation']:
                    self.list_IDs.append(idx)
            else:
                raise AssertionError('{} does not exists'.format(os.path.join(self.image_dir, self.annos[idx]['img_paths'])))

        if train:
            self.transf = transforms.Compose([# transforms.Resize(input_size),
                                              # transforms.RandomAffine(shear=0.05),
                                              # transforms.RandomCrop(input_size),
                                              transforms.RandomResizedCrop(input_size, scale=(0.8, 1.2)),
                                              transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                                              transforms.RandomHorizontalFlip(), # transforms.RandomVerticalFlip(),
                                              transforms.ToTensor(),
                                              ])
        else:
            self.transf = transforms.Compose([transforms.Resize(input_size),
                                              # transforms.CenterCrop(input_size),
                                              transforms.ToTensor(),
                                              ])

        if not self.train:
            self.list_IDs = self.list_IDs[:100]

    def __len__(self):
        """Return number of examples in the dataset"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Generates one sample from dataset"""
        idx = self.list_IDs[index]
        image_name = self.annos[idx]['img_paths']
        assert os.path.exists(os.path.join(self.image_dir, image_name))
        image = Image.open(os.path.join(self.image_dir, image_name))
        joints = self.annos[idx]['joint_self']
        # image = self.transform_images(image, joints)
        image = self.transf(image)

        if self.style_dataset is None:
            style_dataset = 0
        else:
            style_image = self.style_dataset.iterate()
            style_image = self.transf(style_image)
        return image, style_image

    def transform_images(self, image, joints):
        """Crop and preprocess image"""
        # crop image to the person box ++
        min_locs = [1e10, 1e10]
        max_locs = [0, 0]
        for joint in joints:
            joint = [int(i) for i in joint]
            if joint[-1]:
                for i in range(len(min_locs)):
                    min_locs[i] = int(round(min(min_locs[i], joint[i])))
                    max_locs[i] = int(round(max(max_locs[i], joint[i])))

        w = abs(min_locs[0] - max_locs[0])
        h = abs(min_locs[1] - max_locs[1])
        d = max(h, w)
        d //= 1.5

        xc = min_locs[0] + w // 2
        yc = min_locs[1] + h // 2
        x0 = max(0, xc - d)
        x1 = min(image.size[0], xc + d)
        y0 = max(0, yc - d)
        y1 = min(image.size[1], yc + d)

        image = image.crop((x0, y0, x1, y1))
        image_sz = max(image.size[0], image.size[1])

        crop_size = int(round(0.75 * image_sz))

        # random crop and resize first image
        image = self.transf(image)
        return image