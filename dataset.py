import jittor.dataset as data
from PIL import Image
import os


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions: list of allowed extensions
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def make_dataset(dir, extensions):
    """Creates a dataset from the given directory
    Args:
        dir: path to dataset
        extensions: extensions of images to load
    Returns:
        dataset: dataset
    """
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)
    return images


def default_loader(path):
    """open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)"""
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class DatasetFolder(data.Dataset):
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        """
        Args:
            root: root directory of the dataset
            loader: loader of the dataset, see the function default_loader in dataset.py
            extensions: extensions of images to load
            transform:
            target_transform:
        """
        super(DatasetFolder, self).__init__()
        samples = make_dataset(root, extensions)
        if len(samples) == 0:
            raise (
                RuntimeError(
                    "Found 0 files in subfolders of: " + root + "\n"
                    "Supported extensions are: " + ",".join(extensions)
                )
            )
        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        """return number of samples in the dataset"""
        return len(self.samples)


IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif"]


class ImageFolder(DatasetFolder):
    def __init__(
        self, root, transform=None, target_transform=None, loader=default_loader
    ):
        super(ImageFolder, self).__init__(
            root,
            loader,
            IMG_EXTENSIONS,
            transform=transform,
            target_transform=target_transform,
        )
        self.imgs = self.samples
