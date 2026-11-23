from os import listdir
from os.path import isfile, join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (
    Compose,
    RandomCrop,
    ToTensor,
    ToPILImage,
    CenterCrop,
    Resize,
)


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def list_images(dataset_dir):
    """Return the image files in ``dataset_dir`` in a stable (sorted) order."""
    return [join(dataset_dir, x) for x in sorted(listdir(dataset_dir)) if is_image_file(x)]


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose(
        [
            RandomCrop(crop_size),
            ToTensor(),
        ]
    )


def train_lr_transform(crop_size, upscale_factor):
    return Compose(
        [
            ToPILImage(),
            Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
            ToTensor(),
        ]
    )


def display_transform():
    return Compose([ToPILImage(), Resize(400), CenterCrop(400), ToTensor()])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = list_images(dataset_dir)
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(
            Image.open(self.image_filenames[index]).convert("RGB")
        )
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = list_images(dataset_dir)

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index]).convert("RGB")
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = join(dataset_dir, "SRF_" + str(upscale_factor), "data")
        self.hr_path = join(dataset_dir, "SRF_" + str(upscale_factor), "target")
        self.upscale_factor = upscale_factor
        # Images are paired by file name: data/<name> and target/<name>. Only the
        # target (HR) image is read; the LR input is generated from it by bicubic
        # downsampling in __getitem__.
        self.image_names = [x for x in sorted(listdir(self.lr_path)) if is_image_file(x)]
        missing = [x for x in self.image_names if not isfile(join(self.hr_path, x))]
        if missing:
            raise FileNotFoundError(
                "No image with the same name in %s for: %s"
                % (self.hr_path, ", ".join(missing))
            )

    def __getitem__(self, index):
        image_name = self.image_names[index]
        hr_image = Image.open(join(self.hr_path, image_name)).convert("RGB")

        w, h = hr_image.size
        w_crop_size = calculate_valid_crop_size(w, self.upscale_factor)
        h_crop_size = calculate_valid_crop_size(h, self.upscale_factor)
        lr_scale = Resize(
            (h_crop_size // self.upscale_factor, w_crop_size // self.upscale_factor),
            interpolation=Image.BICUBIC,
        )
        hr_image = CenterCrop((h_crop_size, w_crop_size))(hr_image)
        lr_image = lr_scale(hr_image)
        hr_scale = Resize((h_crop_size, w_crop_size), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return (
            image_name,
            ToTensor()(lr_image),
            ToTensor()(hr_restore_img),
            ToTensor()(hr_image),
        )

    def __len__(self):
        return len(self.image_names)
