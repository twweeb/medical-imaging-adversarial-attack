from typing import Dict, List
import argparse
import csv
import time
from dataset import *
from utils import load_model
import numpy as np
import random
from torch import nn
from torchattacks import *


class NoAttack(nn.Module):
    def __init__(self, model=None):
        super().__init__()
        self.model = model

    def forward(self, inputs, labels):
        return inputs


def arg_parsing():
    parser = argparse.ArgumentParser(
        description='Model Robustness Evaluation')
    parser.add_argument('attacks', metavar='attack', type=str, nargs='+',
                        help='attack names')

    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='directory of dataset')
    parser.add_argument('--ground_truth', type=str,
                        help='ground truth of dataset')
    parser.add_argument('--arch', type=str, required=True,
                        help='architecture of model used')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='number of examples/minibatch')
    parser.add_argument('--num_batches', type=int, required=False,
                        help='number of batches (default entire dataset)')
    parser.add_argument('--checkpoint', type=str, default=None, required=True,
                        help='pretrained-model path')
    parser.add_argument('--message', type=str, default="",
                        help='csv message before result')
    parser.add_argument('--per_example', action='store_true', default=False,
                        help='output per-example accuracy')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed')
    parser.add_argument('--output', type=str, help='output CSV')

    args_pool = parser.parse_args()
    return args_pool


if __name__ == '__main__':
    args = arg_parsing()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    num_img, class_names_list, img_filename_list, img_label_list = load_from_class_folder(args.dataset_dir)
    data_loaders, dataset_sizes = load_datasets(img_filename_list, img_label_list, batch_size=args.batch_size)

    # Specify the network to be used
    model = load_model(arch=args.arch, num_classes=len(class_names_list))
    if model is None:
        raise ValueError(f'There is no required architecture {args.arch}.')

    # Load Pre-trained Model
    model = model.cuda()
    checkpoint = torch.load(args.checkpoint)
    try:
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint['model_state_dict'])
            cur_epoch = checkpoint['epoch']+1
            best_acc = checkpoint['best_acc'] if 'best_acc' in checkpoint else 0.0
        else:
            model.load_state_dict(checkpoint)
    except AttributeError:
        model = checkpoint

    # Set the model to the evaluation mode.
    model.eval()

    attack_names: List[str] = args.attacks
    attacks = [eval(attack_name) for attack_name in attack_names]

    batches_correct: Dict[str, List[torch.Tensor]] = \
        {attack_name: [] for attack_name in attack_names}
    batches_ori_correct: Dict[str, List[torch.Tensor]] = \
        {attack_name: [] for attack_name in attack_names}
    batches_time_used: Dict[str, List[torch.Tensor]] = \
        {attack_name: [] for attack_name in attack_names}

    for batch_index, (inputs, labels) in enumerate(data_loaders['test']):
        print(f'BATCH {batch_index:05d}')

        if (
                args.num_batches is not None and
                batch_index >= args.num_batches
        ):
            break

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        for attack_name, attack in zip(attack_names, attacks):
            batch_tic = time.perf_counter()
            adv_inputs = attack(inputs, labels)
            with torch.no_grad():
                ori_logits = model(inputs)
                adv_logits = model(adv_inputs)
            batch_ori_correct = (ori_logits.argmax(1) == labels).detach()
            batch_correct = (adv_logits.argmax(1) == labels).detach()

            batch_accuracy = batch_correct.float().mean().item()
            batch_attack_success_rate = 1.0 - batch_correct[batch_ori_correct].float().mean().item()
            batch_toc = time.perf_counter()
            time_used = torch.tensor(batch_toc - batch_tic)
            print(f'ATTACK {attack_name}',
                  f'accuracy = {batch_accuracy * 100:.1f}',
                  f'attack_success_rate = {batch_attack_success_rate * 100:.1f}',
                  f'time_usage = {time_used:0.2f} s',
                  sep='\t')
            batches_ori_correct[attack_name].append(batch_ori_correct)
            batches_correct[attack_name].append(batch_correct)
            batches_time_used[attack_name].append(time_used)

    print('OVERALL')
    accuracies = []
    attack_success_rates = []
    total_time_used = []
    ori_correct: Dict[str, torch.Tensor] = {}
    attacks_correct: Dict[str, torch.Tensor] = {}
    for attack_name in attack_names:
        ori_correct[attack_name] = torch.cat(batches_ori_correct[attack_name])
        attacks_correct[attack_name] = torch.cat(batches_correct[attack_name])
        accuracy = attacks_correct[attack_name].float().mean().item()
        attack_success_rate = 1.0 - attacks_correct[attack_name][ori_correct[attack_name]].float().mean().item()
        time_used = sum(batches_time_used[attack_name]).item()
        print(f'ATTACK {attack_name}',
              f'accuracy = {accuracy * 100:.1f}',
              f'attack_success_rate = {attack_success_rate * 100:.1f}',
              f'time_usage = {time_used:0.2f} s',
              sep='\t')
        accuracies.append(accuracy)
        attack_success_rates.append(attack_success_rate)
        total_time_used.append(time_used)

    with open(args.output, 'a+') as out_file:
        out_csv = csv.writer(out_file)
        out_csv.writerow([args.message])
        out_csv.writerow(['attack_setting']+attack_names)
        if args.per_example:
            for example_correct in zip(*[
                attacks_correct[attack_name] for attack_name in attack_names
            ]):
                out_csv.writerow(
                    [int(attack_correct.item()) for attack_correct
                     in example_correct])
        out_csv.writerow(['accuracies']+accuracies)
        out_csv.writerow(['attack_success_rates']+attack_success_rates)
        out_csv.writerow(['time_usage']+total_time_used)
        out_csv.writerow(['batch_size', args.batch_size])
        out_csv.writerow(['num_batches', args.num_batches])
        out_csv.writerow([''])
