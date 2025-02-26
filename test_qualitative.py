#
import argparse
import os
import pickle
import pprint
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from model.model_factory import get_model
from parameters import parser

# from test import *
import test as test
from dataset_with_mask import CompositionMaskDataset
from utils import *
import csv

torch.multiprocessing.set_sharing_strategy('file_system')

def predict_logits(model, dataset, config):
    """Function to predict the cosine similarities between the
    images and the attribute-object representations. The function
    also returns the ground truth for attributes, objects, and pair
    of attribute-objects.

    Args:
        model (nn.Module): the model
        text_rep (nn.Tensor): the attribute-object representations.
        dataset (CompositionDataset): the composition dataset (validation/test)
        device (str): the device (either cpu/cuda:0)
        config (argparse.ArgumentParser): config/args

    Returns:
        tuple: the logits, attribute labels, object labels,
            pair attribute-object labels
    """
    model.eval()
    all_attr_gt, all_obj_gt, all_pair_gt = (
        [],
        [],
        [],
    )
    predit_pair = []
    attr2idx = dataset.attr2idx
    obj2idx = dataset.obj2idx
    # print(text_rep.shape)
    pairs_dataset = dataset.pairs
    pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in pairs_dataset]).cuda()
    dataloader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers)
    all_logits = torch.Tensor()
    i = 0
    with torch.no_grad():
        for idx, data in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Testing"
        ):
            # batch_img = data[0].cuda()
            predict = model(data, pairs)
            logits = model.logit_infer(predict, pairs)
            attr_truth, obj_truth, pair_truth = data[2], data[3], data[4]
            predit_pair.append(torch.argmax(logits.softmax(dim=1), dim=1))
            logits = logits.cpu()
            all_logits = torch.cat([all_logits, logits], dim=0)
            all_attr_gt.append(attr_truth)
            all_obj_gt.append(obj_truth)
            all_pair_gt.append(pair_truth)
            # i += 1
            # if i == 10:
            #     break
            # break

    all_attr_gt, all_obj_gt, all_pair_gt = (
        torch.cat(all_attr_gt).to("cpu"),
        torch.cat(all_obj_gt).to("cpu"),
        torch.cat(all_pair_gt).to("cpu"),
    )
    predit_pair = torch.cat(predit_pair).to("cpu")

    return predit_pair, all_attr_gt, all_obj_gt, all_pair_gt


def evaluate(model, dataset, config):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
        model, dataset, config)
    test_stats = test.test(
        dataset,
        evaluator,
        all_logits,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        config
    )
    test_saved_results = dict()
    result = ""
    key_set = ["best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"]
    for key in key_set:
        result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
        test_saved_results[key] = round(test_stats[key], 4)
    print(result)
    test_saved_results['loss'] = loss_avg
    return test_saved_results


if __name__ == "__main__":
    config = parser.parse_args()
    if config.yml_path:
        load_args(config.yml_path, config)
    print(config)
    # set the seed value
    set_seed(config.seed)

    dataset_path = config.dataset_path

    train_dataset = CompositionMaskDataset(args=config,
                                           root=dataset_path,
                                           phase='train',
                                           split='compositional-split-natural',
                                           open_world=config.open_world)

    test_dataset = CompositionMaskDataset(args=config,
                                          root=dataset_path,
                                          phase='test',
                                          split='compositional-split-natural',
                                          open_world=config.open_world)

    allattrs = train_dataset.attrs
    allobj = train_dataset.objs
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    offset = len(attributes)

    model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()
    model.load_state_dict(torch.load(os.path.join(
        "logs_wo_PEM_ut", "final_model.pt"
    )))
    # test_result = evaluate(model, test_dataset, config)
    model.eval()
    predit_pair, all_attr_gt, all_obj_gt, all_pair_gt = predict_logits(
        model, test_dataset, config)

    right_files, false_files, false_pairs = [], [], []
    for idx, pair in enumerate(predit_pair):
        if all_pair_gt[idx] == pair:
            right_files.append(test_dataset.data[idx])
        else:
            false_files.append(test_dataset.data[idx])
            false_pairs.append(test_dataset.pairs[pair])

    with open('right_files_woPEM_ut.csv', 'w', newline='') as f:
        csvwriter = csv.writer(f)
        for row in right_files:
            csvwriter.writerow(row)

    with open('false_files_woPEM_ut.csv', 'w', newline='') as f1:
        csvwriter1 = csv.writer(f1)
        for item1, item2 in zip(false_files, false_pairs):
            csvwriter1.writerow(item1 + list(item2))

    print("done!")
