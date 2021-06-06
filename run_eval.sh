#!/bin/bash

python3 evaluate.py --arch resnet50 \
        --dataset_dir dataset/MedNIST/ \
        --checkpoint pretrained_model/resnet50/mednist.pt   \
        --message resnet50 \
        --batch_size 256 --output results/results.csv \
        "NoAttack()" \
        "FGSM(model, eps=8/255)" \
        "BIM(model, eps=8/255, alpha=2/255, steps=20)" \
        "RFGSM(model, eps=8/255, alpha=4/255, steps=20)" \
        "PGD(model, eps=8/255, alpha=2/255, steps=20)" \
        "FFGSM(model, eps=8/255, alpha=12/255)" \
        "TPGD(model, eps=8/255, alpha=2/255, steps=20)" \
        "MIFGSM(model, eps=8/255, decay=1.0, steps=20)" \
        "PGDDLR(model, eps=8/255, alpha=2/255, steps=20)"
