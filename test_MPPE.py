#
import argparse
import os
import pickle
import pprint


import numpy as np
import torch
import torch.backends.cudnn as cudnn
import copy
import tqdm
import yaml
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from model.model_factory import get_model
from parameters import parser

from dataset_with_mask import CompositionMaskDataset
from utils import *
from test import Evaluator, threshold_with_feasibility, test

cudnn.benchmark = True

torch.multiprocessing.set_sharing_strategy('file_system')
device = "cuda" if torch.cuda.is_available() else "cpu"

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
    loss = 0
    with torch.no_grad():
        for idx, data in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Testing"
        ):
            # batch_img = data[0].cuda()
            predict = model(data, pairs)
            logits = model.logit_infer(predict, pairs)
            loss += model.loss_calu(predict, data).item()
            # attr_truth, obj_truth, pair_truth = data[2], data[3], data[4]
            attr_truth, obj_truth, pair_truth = data[2], data[3], data[4]
            logits = logits.cpu()
            all_logits = torch.cat([all_logits, logits], dim=0)
            all_attr_gt.append(attr_truth)
            all_obj_gt.append(obj_truth)
            all_pair_gt.append(pair_truth)

            # break

    all_attr_gt, all_obj_gt, all_pair_gt = (
        torch.cat(all_attr_gt).to("cpu"),
        torch.cat(all_obj_gt).to("cpu"),
        torch.cat(all_pair_gt).to("cpu"),
    )

    return all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss / len(dataloader)


def predict_logits_text_first(model, dataset, config):
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
    loss = 0
    with torch.no_grad():
        # Composition and PE features depend on each candidate pair.
        # Attribute/object features cover the full vocabulary in every chunk.
        text_feats = [[], None, None, []]
        for start in range(0, len(pairs), config.text_encoder_batch_size):
            cur_pair = pairs[start:start + config.text_encoder_batch_size]
            features = model.encode_text_for_open(cur_pair)
            text_feats[0].append(features[0])
            text_feats[3].append(features[3])
            if text_feats[1] is None:
                text_feats[1], text_feats[2] = features[1], features[2]
        text_feats[0] = torch.cat(text_feats[0], dim=0)
        text_feats[3] = torch.cat(text_feats[3], dim=0)
        for idx, data in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Testing"
        ):
            # batch_img = data[0].cuda()
            predict = model.forward_for_open(data, text_feats)
            logits = model.logit_infer(predict, pairs)
            loss += model.loss_calu(predict, data).item()
            # attr_truth, obj_truth, pair_truth = data[2], data[3], data[4]
            attr_truth, obj_truth, pair_truth = data[2], data[3], data[4]
            logits = logits.cpu()
            all_logits = torch.cat([all_logits, logits], dim=0)
            all_attr_gt.append(attr_truth)
            all_obj_gt.append(obj_truth)
            all_pair_gt.append(pair_truth)
            # break

    all_attr_gt, all_obj_gt, all_pair_gt = (
        torch.cat(all_attr_gt).to("cpu"),
        torch.cat(all_obj_gt).to("cpu"),
        torch.cat(all_pair_gt).to("cpu"),
    )

    # ? delete the text encoder to save CUDA memory
    # del model.transformer
    # torch.cuda.empty_cache()

    return all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss / len(dataloader)


if __name__ == "__main__":
    config = parser.parse_args()
    if config.yml_path:
        load_args(config.yml_path, config)
    # Explicit CLI options override YAML; unspecified options retain YAML values.
    config = parser.parse_args(namespace=config)
    if not config.load_model:
        parser.error("--load_model is required (e.g. logs/val_best.pt)")
    if not os.path.isfile(config.load_model):
        parser.error(f"Checkpoint not found: {config.load_model}")
    if config.text_encoder_batch_size <= 0 or config.threshold_trials <= 0:
        parser.error("text_encoder_batch_size and threshold_trials must be positive")
    print(config)
    # set the seed value
    set_seed(config.seed)
    print("----")
    test_type = 'OPEN WORLD' if config.open_world else 'CLOSED WORLD'
    print(f"{test_type} evaluation details")
    print("----")
    print(f"dataset: {config.dataset}")

    dataset_path = config.dataset_path
    print('loading validation dataset')
    val_dataset = CompositionMaskDataset(args=config,
                                         root=dataset_path,
                                     phase='val',
                                     split='compositional-split-natural',
                                     open_world=config.open_world)

    allattrs = val_dataset.attrs
    allobj = val_dataset.objs
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    offset = len(attributes)

    model = get_model(config, attributes=attributes, classes=classes, offset=offset).cuda()
    model.load_state_dict(torch.load(config.load_model, map_location="cpu"), strict=True)

    predict_logits_func = predict_logits
    # ? can be deleted if not needed
    if (hasattr(config, 'text_first') and config.text_first):
        print('text first')
        predict_logits_func = predict_logits_text_first

    print('evaluating on the validation set')
    if config.open_world and config.threshold is None:
        evaluator = Evaluator(val_dataset, model=None)
        feasibility_path = os.path.join(
            DIR_PATH, f'data/feasibility_{config.dataset}.pt')
        unseen_scores = torch.load(
            feasibility_path,
            map_location='cpu')['feasibility']
        seen_mask = val_dataset.seen_mask.to('cpu')
        min_feasibility = (unseen_scores + seen_mask * 10.).min()
        max_feasibility = (unseen_scores - seen_mask * 10.).max()
        thresholds = np.linspace(
            min_feasibility,
            max_feasibility,
            num=config.threshold_trials)
        best_auc = -float("inf")
        best_th = -10
        val_stats = None
        with torch.no_grad():
            all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = predict_logits_func(
                model, val_dataset, config)
            for th in thresholds:
                temp_logits = threshold_with_feasibility(
                    all_logits, val_dataset.seen_mask, threshold=th, feasiblity=unseen_scores)
                results = test(
                    val_dataset,
                    evaluator,
                    temp_logits,
                    all_attr_gt,
                    all_obj_gt,
                    all_pair_gt,
                    config
                )
                auc = results['AUC']
                if auc > best_auc:
                    best_auc = auc
                    best_th = th
                    print('New best AUC', best_auc)
                    print('Threshold', best_th)
                    val_stats = copy.deepcopy(results)
    else:
        best_th = config.threshold
        evaluator = Evaluator(val_dataset, model=None)
        if config.open_world:
            feasibility_path = os.path.join(
                DIR_PATH, f'data/feasibility_{config.dataset}.pt')
            unseen_scores = torch.load(
                feasibility_path,
                map_location='cpu')['feasibility']
        with torch.no_grad():
            all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = predict_logits_func(
                model, val_dataset, config)
            if config.open_world:
                print('using threshold: ', best_th)
                all_logits = threshold_with_feasibility(
                    all_logits, val_dataset.seen_mask, threshold=best_th, feasiblity=unseen_scores)
            results = test(
                val_dataset,
                evaluator,
                all_logits,
                all_attr_gt,
                all_obj_gt,
                all_pair_gt,
                config
            )
        val_stats = copy.deepcopy(results)
        result = ""
        for key in val_stats:
            result = result + key + "  " + str(round(val_stats[key], 4)) + "| "
        print(result)

    print('loading test dataset')
    test_dataset = CompositionMaskDataset(args=config,
                                          root=dataset_path,
                                          phase='test',
                                          split='compositional-split-natural',
                                          open_world=config.open_world)





    # test_result = evaluate(model, test_dataset, config)
    print('evaluating on the test set')
    with torch.no_grad():
        evaluator = Evaluator(test_dataset, model=None)
        all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = predict_logits_func(
            model, test_dataset, config)
        if config.open_world and best_th is not None:
            print('using threshold: ', best_th)
            all_logits = threshold_with_feasibility(
                all_logits,
                test_dataset.seen_mask,
                threshold=best_th,
                feasiblity=unseen_scores)
        test_stats = test(
            test_dataset,
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config
        )

        result = ""
        for key in test_stats:
            result = result + key + "  " + \
                str(round(test_stats[key], 4)) + "| "
        print(result)

    results = {
        'val': val_stats,
        'test': test_stats,
    }

    print("done!")
