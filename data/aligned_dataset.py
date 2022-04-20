import os
from data.base_dataset import BaseDataset
from glob import glob
import cv2
import numpy as np
import albumentations as A
from tifffile import imread


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        # self.filenames = sorted(glob(os.path.join(self.dir_AB, 'DEM/*')))
        self.filenames = sorted(glob(os.path.join(self.dir_AB, 'inputs/*')))
        self.filenames = self.filenames[:min(len(self.filenames), opt.max_dataset_size)]
        # assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc
        self.aug = A.Compose(
            [
                A.VerticalFlip(p=0.3),
                A.HorizontalFlip(p=0.3),
                A.Transpose(p=0.3),
                A.RandomRotate90(p=0.3),
                A.RandomCrop(width=self.opt.crop_size, height=self.opt.crop_size, p=1.0)
            ]
        )

    def normalize(self, image):
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + np.finfo(np.float32).eps)
        if image.ndim == 2:
            image = np.expand_dims(image, -1)
        return image

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        filename = self.filenames[index]
        output_filename = filename.replace('/inputs/', '/outputs/')
        # output_filenames = [filename.replace('/DEM/', raster) for raster in ['/LD/', '/SLRM/', '/SLOPE/', '/SVF64/']]
        # A = self.normalize(cv2.imread(filename, cv2.IMREAD_UNCHANGED))
        A = imread(filename)
        # B = [self.normalize(cv2.imread(f, cv2.IMREAD_UNCHANGED)) for f in output_filenames]
        # B = np.concatenate(B, -1)
        B = imread(output_filename)

        if self.opt.phase == 'train':
            augmented = self.aug(image=A, mask=B)
            A = augmented['image']
            B = augmented['mask']

        if A.ndim == 2:
            A = np.expand_dims(A, -1)
        A = A.transpose((2, 0, 1))

        if B.ndim == 2:
            B = np.expand_dims(B, -1)
        B = B.transpose((2, 0, 1))

        # return {'A': A, 'B': B, 'A_paths': filename, 'B_paths': output_filenames}
        return {'A': A, 'B': B, 'A_paths': filename, 'B_paths': output_filename}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.filenames)
