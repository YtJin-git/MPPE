# Python imports
from tqdm import tqdm
from os.path import join as ospj
import argparse
#Local imports
import czsl_dataset as dset_clip
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from tqdm import tqdm
from local_utils.log_util import init_logger
from local_utils.config_utils import parse_config_utils
from models import build_sam_clip_text_ins_segmentor

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image_path', type=str, default='/remote-home/CZSL/test.png')
    parser.add_argument('--insseg_cfg_path', type=str, default='./config/insseg.yaml')
    parser.add_argument('--text', type=str, default='deer')
    parser.add_argument('--cls_score_thresh', type=float, default=None)
    parser.add_argument('--save_dir', type=str, default='./output/insseg')
    parser.add_argument('--use_text_prefix', action='store_true')

    return parser.parse_args()

def seg(dataset1, dataset_name, phase='train'):
        dataset1 = tqdm(dataset1, desc='|--Training')
        os.makedirs(save_root_path, exist_ok=True)
        for idx, data in enumerate(dataset1):
                image, attr, obj = data
                if dataset_name == 'mit-states':
                        pair, img = image.split('/')
                        pair = pair.replace('_', ' ')
                        image = pair + '/' + img
                        os.makedirs(save_root_path+'/'+pair, exist_ok=True)
                elif dataset_name == 'ut-zap50k':
                    pair, img = image.split('/')
                    os.makedirs(save_root_path + '/' + pair, exist_ok=True)
                unique_labels[0] = obj
                image_path = ospj(data_root, dataset_name, 'images', image)
                if phase == 'train':
                    ret = segmentor.seg_image(image_path, unique_label=unique_labels, use_text_prefix=use_text_prefix)
                else:
                    ret = segmentor.seg_image(image_path, unique_label=None, use_text_prefix=use_text_prefix)
                mask_save_path = ospj(save_root_path, image)
                cv2.imwrite(mask_save_path, ret['ins_seg_mask'])


args = init_args()
use_text_prefix = True if args.use_text_prefix else False
dataset = 'cgqa'
data_root = '/home/datasets'
splitname = 'compositional-split-natural'
save_root_path = '/home/datasets/cgqa/masks'
insseg_cfg_path = args.insseg_cfg_path
insseg_cfg = parse_config_utils.Config(config_path=insseg_cfg_path)
segmentor = build_sam_clip_text_ins_segmentor(cfg=insseg_cfg)
if args.text is not None:
        unique_labels = args.text.split(',')
else:
        unique_labels = None

trainset = dset_clip.CompositionDataset(
        args=None,
        root=ospj(data_root, dataset),
        phase='train',
        split=splitname,
        open_world=False
    )
valset = dset_clip.CompositionDataset(
        args=None,
        root=ospj(data_root, dataset),
        phase='val',
        split=splitname,
        open_world=False
    )
testset = dset_clip.CompositionDataset(
        args=None,
        root=ospj(data_root, dataset),
        phase='test',
        split=splitname,
        open_world=False
    )

print("Trainset processing")
seg(trainset, dataset, 'train')
print("Valset processing")
seg(valset, dataset, 'val')
print("Testset processing")
seg(testset, dataset, 'test')


print("Finished")



