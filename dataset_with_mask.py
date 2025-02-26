from itertools import product

import numpy as np
from random import choice
from os.path import join as ospj
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, RandomHorizontalFlip,
                                    RandomPerspective, RandomRotation, Resize,
                                    ToTensor)
from torchvision.transforms.transforms import RandomResizedCrop

BICUBIC = InterpolationMode.BICUBIC
n_px = 224


def transform_image(split="train", imagenet=False):
    if imagenet:
        # from czsl repo.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = Compose(
            [
                RandomResizedCrop(n_px, antialias=True),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(
                    mean,
                    std,
                ),
            ]
        )
        return transform

    if split == "test" or split == "val":
        transform = Compose(
            [
                Resize(n_px, interpolation=BICUBIC, antialias=True),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        transform = Compose(
            [
                # RandomResizedCrop(n_px, interpolation=BICUBIC),
                Resize(n_px, interpolation=BICUBIC, antialias=True),
                CenterCrop(n_px),
                RandomHorizontalFlip(),
                RandomPerspective(),
                RandomRotation(degrees=5),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    return transform


class ImageLoader:
    def __init__(self, root):
        self.root_dir = root

    def __call__(self, img):
        img = Image.open(ospj(self.root_dir, img)).convert('RGB')  # We don't want alpha
        return img


class MaskLoader:
    def __init__(self, root):
        self.root_dir = root

    def __call__(self, mask):
        mask = np.array(Image.open(ospj(self.root_dir, mask)))
        if len(mask.shape) == 2: binary_mask = (mask == 255)
        if len(mask.shape) == 3: binary_mask = (mask[:, :, 0] == 255)
        return binary_mask


class CompositionMaskDataset(Dataset):
    def __init__(
            self,
            args,
            root,
            phase,
            split='compositional-split-natural',
            open_world=False,
            update_image_features=False,
            imagenet=False
            # inductive=True
    ):
        self.args = args
        self.root = root
        self.phase = phase
        self.split = split
        self.open_world = open_world
        self.update_image_features = update_image_features

        # new addition
        # if phase == 'train':
        #     self.inductive = inductive
        # else:
        #     self.inductive = False

        self.feat_dim = None
        self.transform = transform_image(phase, imagenet=imagenet)
        self.mask_transform = Compose([ToTensor(),
            Resize((224, 224), antialias=True), # change to (336,336) when using ViT-L/14@336px
            Normalize(0.5, 0.26)
            ])
        self.loader = ImageLoader(ospj(self.root, 'images'))
        self.mask_loader = MaskLoader(ospj(self.root, 'masks'))

        self.attrs, self.objs, self.pairs, \
                self.train_pairs, self.val_pairs, \
                self.test_pairs = self.parse_split()

        if self.open_world:
            self.pairs = list(product(self.attrs, self.objs))

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))

        self.train_pair_to_idx = dict(
            [(pair, idx) for idx, pair in enumerate(self.train_pairs)]
        )

        if self.open_world:
            mask = [1 if pair in set(self.train_pairs) else 0 for pair in self.pairs]
            self.seen_mask = torch.BoolTensor(mask) * 1.

            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.train_pairs:
                self.obj_by_attrs_train[a].append(o)

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.train_pairs:
                self.attrs_by_obj_train[o].append(a)

    def get_split_info(self):
        data = torch.load(ospj(self.root, 'metadata_{}.t7'.format(self.split)))
        train_data, val_data, test_data = [], [], []
        for instance in data:
            image, attr, obj, settype = instance['image'], instance[
                'attr'], instance['obj'], instance['set']

            if attr == 'NA' or (attr,
                                obj) not in self.pairs or settype == 'NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = [image, attr, obj]
            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                # pairs = [t.split() if not '_' in t else t.split('_') for t in pairs]
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            ospj(self.root, self.split, 'train_pairs.txt')
        )
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            ospj(self.root, self.split, 'val_pairs.txt')
        )
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            ospj(self.root, self.split, 'test_pairs.txt')
        )

        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def load_img(self, image):
        # Decide what to output

        # if not self.update_features:
        #     img = self.activations[image]
        # else:
        img = self.loader(image)
        img = self.transform(img)

        return img

    def load_mask(self, mask):
        # Decide what to output
        if self.args.dataset == 'cgqa':
            mask = self.mask_loader(mask)
            mask = self.mask_transform((mask * 255).astype(np.uint8))
        else:
            # 全黑mask
            mask = self.mask_loader(mask)
            zero_mask = np.zeros(mask.shape, dtype=bool)
            zero_mask = self.mask_transform((zero_mask * 255).astype(np.uint8))
            mask = zero_mask
        return mask

    def revise_image_path(self, image):
        if self.args.dataset == 'mit-states':
            pair, img = image.split('/')
            pair = pair.replace('_', ' ')
            image = pair + '/' + img
        return image

    def __getitem__(self, index):
        image, attr, obj = self.data[index]
        image = self.revise_image_path(image)
        img = self.load_img(image)
        mask = self.load_mask(image)

        if self.phase == 'train':
            data = [
                img, mask, self.attr2idx[attr], self.obj2idx[obj], self.train_pair_to_idx[(attr, obj)]
            ]

        else:
            data = [
                img, mask, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]
            ]

        return data

    def __len__(self):
        return len(self.data)



