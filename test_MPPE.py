#
import argparse
import os
import pickle
import pprint

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

# def train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset):
#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=config.train_batch_size,
#         shuffle=True,
#         num_workers=config.num_workers
#     )
#
#     model.train()
#     best_val_metric = 0
#     best_test_metric = 0
#     best_val_loss = 1e5
#     best_test_loss = 1e5
#     best_epoch = 0
#     final_model_state = None
#
#     val_results = []
#     test_results = []
#
#     scheduler = get_scheduler(optimizer, config, len(train_dataloader))
#     attr2idx = train_dataset.attr2idx
#     obj2idx = train_dataset.obj2idx
#
#     train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
#                                 for attr, obj in train_dataset.train_pairs]).cuda()
#
#     train_losses = []
#
#     for i in range(config.epoch_start, config.epochs):
#         progress_bar = tqdm.tqdm(
#             total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
#         )
#
#         epoch_train_losses = []
#         for bid, batch in enumerate(train_dataloader):
#             predict = model(batch, train_pairs)
#
#             loss = model.loss_calu(predict, batch)
#
#             # normalize loss to account for batch accumulation
#             loss = loss / config.gradient_accumulation_steps
#
#             # backward pass
#             loss.backward()
#
#             # weights update
#             if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
#                 optimizer.step()
#                 optimizer.zero_grad()
#             scheduler = step_scheduler(scheduler, config, bid, len(train_dataloader))
#
#             epoch_train_losses.append(loss.item())
#             progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})
#             progress_bar.update()
#             # break
#
#         progress_bar.close()
#         progress_bar.write(f"epoch {i + 1} "
#                            f"train loss {np.mean(epoch_train_losses)}")
#         train_losses.append(np.mean(epoch_train_losses))
#
#         # wandb.log({"train_loss": np.mean(epoch_train_losses)})
#
#         if (i + 1) % config.save_every_n == 0:
#             torch.save(model.state_dict(), os.path.join(config.save_path, f"epoch_{i}.pt"))
#
#         print("Evaluating val dataset:")
#         val_result = evaluate(model, val_dataset, config)
#         val_results.append(val_result)
#
#         if config.val_metric == 'best_loss' and val_result['loss'] < best_val_loss:
#             best_val_loss = val_result['loss']
#             best_epoch = i
#             torch.save(model.state_dict(), os.path.join(
#                 config.save_path, "val_best.pt"))
#         if config.val_metric != 'best_loss' and val_result[config.val_metric] > best_val_metric:
#             best_val_metric = val_result[config.val_metric]
#             best_epoch = i
#             torch.save(model.state_dict(), os.path.join(
#                 config.save_path, "val_best.pt"))
#
#         final_model_state = model.state_dict()
#         if i + 1 == config.epochs:
#             print("--- Evaluating test dataset on Closed World ---")
#             model.load_state_dict(torch.load(os.path.join(
#                 config.save_path, "val_best.pt"
#             )))
#             test_result = evaluate(model, test_dataset, config)
#
#     if config.save_final_model:
#         torch.save(final_model_state, os.path.join(config.save_path, f'final_model.pt'))
#
#
# def evaluate(model, dataset, config):
#     model.eval()
#     evaluator = test.Evaluator(dataset, model=None)
#     all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
#         model, dataset, config)
#     test_stats = test.test(
#         dataset,
#         evaluator,
#         all_logits,
#         all_attr_gt,
#         all_obj_gt,
#         all_pair_gt,
#         config
#     )
#     test_saved_results = dict()
#     result = ""
#     key_set = ["best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"]
#     for key in key_set:
#         result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
#         test_saved_results[key] = round(test_stats[key], 4)
#     print(result)
#     test_saved_results['loss'] = loss_avg
#     return test_saved_results


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
            attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]
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
        text_feats = [[], [], []]
        num_text_batch = pairs.shape[0] // config.text_encoder_batch_size
        for i_text_batch in range(num_text_batch):
            cur_pair = pairs[i_text_batch*config.text_encoder_batch_size:(i_text_batch+1)*config.text_encoder_batch_size, :]
            cur_text_feats = model.encode_text_for_open(cur_pair)
            for i_item in range(len(text_feats)):
                text_feats[i_item].append(cur_text_feats[i_item])
        if pairs.shape[0] % config.text_encoder_batch_size != 0:
            cur_pair = pairs[num_text_batch*config.text_encoder_batch_size:, :]
            cur_text_feats = model.encode_text_for_open(cur_pair)
            for i_item in range(len(text_feats)):
                text_feats[i_item].append(cur_text_feats[i_item])
        for i_item in range(len(text_feats)):
            text_feats[i_item] = torch.cat(text_feats[i_item], dim=0)
        for idx, data in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Testing"
        ):
            # batch_img = data[0].cuda()
            predict = model.forward_for_open(data, text_feats)
            logits = model.logit_infer(predict, pairs)
            loss += model.loss_calu(predict, data).item()
            # attr_truth, obj_truth, pair_truth = data[2], data[3], data[4]
            attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]
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
    model.load_state_dict(torch.load(os.path.join(
        "logs_cgqa_ow", "epoch_0.pt"
    )))

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
        best_auc = 0.
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
