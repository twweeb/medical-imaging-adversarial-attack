import argparse
import time
import copy
import torch.nn as nn
from torch.optim import lr_scheduler
from timeit import default_timer as timer
from dataset import *
from utils import load_model
import numpy as np
import random


def arg_parsing():
    parser = argparse.ArgumentParser(
        description='Model Training Setting')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='directory of dataset')
    parser.add_argument('--ground_truth', type=str,
                        help='ground truth of dataset')
    parser.add_argument('--arch', type=str, required=True,
                        help='architecture of model used')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='dir to save of the training model')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='number of examples/minibatch')
    parser.add_argument('--epoch', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='pretrained-model path')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='clip gradients to this value')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed')

    args_pool = parser.parse_args()
    return args_pool


def train_model(model, dataloaders, criterion, optimizer, scheduler, cur_epoch, best_acc=0.0, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(cur_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            start = timer()
            for batch_id, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                print(
                    f'Epoch: {epoch}\t{100 * (batch_id + 1) / len(dataloaders[phase]):.2f}% complete.',
                    f'\t {timer() - start:.2f} seconds elapsed in epoch.',
                    end='\r')

            # In this project, we do not need to adjust the learning rate.
            # if phase == 'train':
            #     scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'\n{phase} | Epoch: {epoch} \tLoss: {epoch_loss:.4f}\t\t Accuracy: {100 * epoch_acc:.2f}%')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_acc': best_acc,
                }, args.model_dir + args.arch + '/model-best.pt')
                print()
                print('Save Best Model!')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def evaluate(_model, phase='test'):
    _model = _model.eval()

    start = timer()
    running_corrects = 0
    for batch_id, (inputs, labels) in enumerate(data_loaders[phase]):
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = _model(inputs).squeeze()
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

        print(
            f'{phase} | {100 * (batch_id + 1) / len(data_loaders[phase]):.2f}% complete.',
            f'\t {timer() - start:.2f} seconds elapsed in testing.',
            end='\r')
    print()

    accuracy = running_corrects.double() / dataset_sizes[phase]
    print(phase.capitalize() + ' Accuracy: {:.2f}%'.format(100 * accuracy))


if __name__ == '__main__':
    args = arg_parsing()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # show_info(args.dataset_dir)
    num_img, class_names_list, img_filename_list, img_label_list = load_from_class_folder(args.dataset_dir)

    # random_sample(img_filename_list, img_label_list, class_names_list)
    data_loaders, dataset_sizes = load_datasets(img_filename_list, img_label_list, batch_size=args.batch_size)

    # Specify the network to be used
    model = load_model(arch=args.arch, num_classes=len(class_names_list))
    if model is None:
        raise ValueError(f'There is no required architecture {args.arch}.')

    model = model.cuda()
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    # optimizer_ft = torch.optim.AdamW(model.parameters(), lr=0.001)
    optimizer_ft = torch.optim.SGD(model.parameters(),
                                   lr=0.001,
                                   momentum=0.9,
                                   weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[5, 15], gamma=0.1)

    torch.cuda.empty_cache()

    cur_epoch = 0
    best_acc = 0.0
    # Load checkpoint if provided

    if args.checkpoint is not None:
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

    model = train_model(model, data_loaders, criterion, optimizer_ft, exp_lr_scheduler, cur_epoch, best_acc=best_acc,
                        num_epochs=args.epoch)

    evaluate(model, phase='test')
