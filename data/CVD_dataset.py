
from PIL import Image
from torch.utils.data import Dataset
import data.util as Util


class CVDDataset(Dataset):
    def __init__(self, dataroot, datatype, split='train', data_len=-1):
        self.datatype = datatype
        self.data_len = data_len
        self.split = split

        if datatype == 'img':
            self.original_path = Util.get_paths_from_images(
                '{}/A'.format(dataroot))
            self.recolor_path = Util.get_paths_from_images(
                '{}/B'.format(dataroot))
            self.dataset_len = len(self.recolor_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_original = None
        img_recolor = None

        img_original = Image.open(self.original_path[index]).convert("RGB")
        img_recolor = Image.open(self.recolor_path[index]).convert("RGB")
        img_style = Image.open(self.recolor_path[index]).convert("RGB")

        img_original = img_original.resize((128, 128), Image.BICUBIC)
        img_recolor = img_recolor.resize((128, 128), Image.BICUBIC)
        img_style = img_style.resize((128, 128), Image.BICUBIC)

        [img_original, img_recolor, img_style] = Util.transform_augment(
        [img_original, img_recolor, img_style], split=self.split, min_max=(-1, 1))
        return {'original': img_original, 'inf': img_recolor, 'style': img_style, 'Index': index}