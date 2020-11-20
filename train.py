import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils import JetVectorDataset, collate_fn, train, test, validate
from model.tagger import NetVLADTagger
from pathlib import Path
parser = argparse.ArgumentParser(description="Train NetVLAD-tagger on specified dataset")
parser.add_argument('--train-data', type=str, metavar='/path/to/train/dataset/*.root', help='path to the train dataset')
parser.add_argument('--test-data', type=str, metavar='/path/to/test/dataset/*.root', help='path to the test dataset')
parser.add_argument('--val-data', type=str, metavar='/path/to/val/dataset/*.root', help='path to the val dataset')
parser.add_argument('--variables', type=str, help='input variable set to use')
parser.add_argument('--clusters', type=int, help='number of clusters to use in NetVLAD', default=33)
parser.add_argument('--depth', type=int, help='number of residual blocks to use', default=2)
parser.add_argument('--signal', type=str, help='select signal in dataset')
parser.add_argument('--track-shuffling', action='store_true', help='enable track shuffling during training')
parser.add_argument('--jobid', type=str, metavar='run index (can be SLURM job id)', default='jetvlad_run_0')
args = parser.parse_args()


def main():
    print("Loading test dataset...", end=" ")
    test_data = JetVectorDataset(args.test_data, signal=args.signal, variables=args.variables, track_shuffling=False)
    print("Done")

    print("Loading val dataset...", end=" ")
    val_data = JetVectorDataset(args.val_data, signal=args.signal, variables=args.variables, track_shuffling=False)
    print("Done")

    print("Loading train dataset...", end=" ")
    train_data = JetVectorDataset(args.train_data, signal=args.signal, variables=args.variables, track_shuffling=args.track_shuffling)
    print("Done")

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False, num_workers=2,
                                              collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=256, shuffle=False, num_workers=2,
                                             collate_fn=collate_fn)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True, num_workers=2,
                                               collate_fn=collate_fn, drop_last=True)

    run_path = Path().resolve() / 'training_runs' / args.jobid
    run_path.mkdir(parents=True, exist_ok=True)

    save_path = run_path / f'netvlad_best_val_loss_{args.train_data.replace(".root", "")}_{args.jobid}.pth'

    if args.variables == 'trkvtxfrag':
        model = NetVLADTagger(args.clusters, 8, args.depth)
    elif args.variables == 'trkfrag':
        model = NetVLADTagger(args.clusters, 6, args.depth)
    elif args.variables == 'trkvtx':
        model = NetVLADTagger(args.clusters, 5, args.depth)
    elif args.variables == 'trk':
        model = NetVLADTagger(args.clusters, 3, args.depth)
    elif args.variables == 'vtx':
        model = NetVLADTagger(args.clusters, 2, args.depth)

    model.cuda()
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.013, momentum=0.9, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=3)

    best_val = np.inf
    best_acc = 0
    best_roc = 0
    early_stop = 0

    for i in range(2000):
        train(train_loader, model, criterion, optimizer, scheduler, i)
        val_loss, acc = validate(val_loader, model, criterion)
        roc = test(val_loader, args.train_data, model, False)

        if best_val - val_loss > 0.001:
            best_val = val_loss
            best_acc = acc
            best_roc = roc
            torch.save(model.state_dict(), save_path.as_posix())
            early_stop = 0
        else:
            early_stop += 1

        print("Best AUC-ROC: " + str(best_roc))
        print("Best acc: " + str(best_acc))
        print("Best val loss: " + str(best_val))

        if early_stop == 10:
            break

    model.load_state_dict(torch.load(save_path.as_posix()))

    log_path = run_path / f'{args.train_data.replace(".root", "")}_{args.jobid}'
    final_roc_auc = test(test_loader,
                         log_path,
                         model,
                         generate_curve=True)
    print(f'Final ROC-AUC: {final_roc_auc}')


if __name__ == "__main__":
    main()