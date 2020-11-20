import time
import torch
import torch.utils.data
import uproot
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc


class JetVectorDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, signal, variables, track_shuffling):
        assert signal in ['c_vs_b', 'hf_vs_l', 'c_vs_all', 'b_vs_all']
        assert variables in ['trk', 'vtx', 'trkvtx', 'trkfrag', 'trkvtxfrag']

        # Since periodical loading from TTree via uproot is slow we load dataset into memory,
        # this might delay start of the training by ~15-30 minutes (depending on the size of the TTree)
        # but resulting run time will be *much* faster
        self.file = uproot.open(file_path)['T']
        self.signal = signal
        self.jets = 0
        self.track_shuffling = track_shuffling
        jets = self.file.arrays(self.file.keys())

        # Calculate normalizations of the observables
        mu_pt, sigma_pt = self._calculate_normalization('fPt', jets)
        mu_eta, sigma_eta = self._calculate_normalization('fEta', jets)
        mu_phi, sigma_phi = self._calculate_normalization('fPhi', jets)
        if variables in ['trkfrag', 'trkvtxfrag']:
            mu_z, sigma_z = self._calculate_normalization('fZ', jets)
            mu_m, sigma_m = self._calculate_normalization('fM', jets)
            mu_deltaR, sigma_deltaR = self._calculate_normalization('fDeltaR', jets)
        if variables in ['vtx', 'trkvtx', 'trkvtxfrag']:
            mu_dca_z, sigma_dca_z = self._calculate_normalization('fDCA_z', jets)
            mu_dca_xy, sigma_dca_xy = self._calculate_normalization('fDCA_xy', jets)

        # Generate matrices representing jet, tracks are in the rows, columns represent different obeservables
        self.X, self.y = list(), list()
        for i in range(len(self.file)):
            if variables == 'trkvtxfrag':
                jet_matrix = np.hstack(
                    ((jets[b'fPt'][i].reshape(-1, 1) - mu_pt) / sigma_pt,
                     (jets[b'fEta'][i].reshape(-1, 1) - mu_eta) / sigma_eta,
                     (jets[b'fPhi'][i].reshape(-1, 1) - mu_phi) / sigma_phi,
                     (jets[b'fZ'][i].reshape(-1, 1) - mu_z) / sigma_z,
                     (jets[b'fDeltaR'][i].reshape(-1, 1) - mu_deltaR) / sigma_deltaR,
                     (jets[b'fZ'][i].reshape(-1, 1) * jets[b'fDeltaR'][i].reshape(-1, 1) * jets[b'fDeltaR'][i].reshape(
                         -1, 1) - mu_m) / sigma_m,
                     (jets[b'fDCA_z'][i].reshape(-1, 1) - mu_dca_z) / sigma_dca_z,
                     (jets[b'fDCA_xy'][i].reshape(-1, 1) - mu_dca_xy) / sigma_dca_xy))
            elif variables == 'trkfrag':
                jet_matrix = np.hstack(
                    ((jets[b'fPt'][i].reshape(-1, 1) - mu_pt) / sigma_pt,
                     (jets[b'fEta'][i].reshape(-1, 1) - mu_eta) / sigma_eta,
                     (jets[b'fPhi'][i].reshape(-1, 1) - mu_phi) / sigma_phi,
                     (jets[b'fZ'][i].reshape(-1, 1) - mu_z) / sigma_z,
                     (jets[b'fDeltaR'][i].reshape(-1, 1) - mu_deltaR) / sigma_deltaR,
                     (jets[b'fZ'][i].reshape(-1, 1) * jets[b'fDeltaR'][i].reshape(-1, 1) * jets[b'fDeltaR'][i].reshape(-1, 1) - mu_m) / sigma_m))
            elif variables == 'trkvtx':
                jet_matrix = np.hstack(
                    ((jets[b'fPt'][i].reshape(-1, 1) - mu_pt) / sigma_pt,
                     (jets[b'fEta'][i].reshape(-1, 1) - mu_eta) / sigma_eta,
                     (jets[b'fPhi'][i].reshape(-1, 1) - mu_phi) / sigma_phi,
                     (jets[b'fDCA_z'][i].reshape(-1, 1) - mu_dca_z) / sigma_dca_z,
                     (jets[b'fDCA_xy'][i].reshape(-1, 1) - mu_dca_xy) / sigma_dca_xy))
            elif variables == 'trk':
                jet_matrix = np.hstack(
                    ((jets[b'fPt'][i].reshape(-1, 1) - mu_pt) / sigma_pt,
                     (jets[b'fEta'][i].reshape(-1, 1) - mu_eta) / sigma_eta,
                     (jets[b'fPhi'][i].reshape(-1, 1) - mu_phi) / sigma_phi))
            elif variables == 'vtx':
                jet_matrix = np.hstack(
                    ((jets[b'fDCA_z'][i].reshape(-1, 1) - mu_dca_z) / sigma_dca_z,
                     (jets[b'fDCA_xy'][i].reshape(-1, 1) - mu_dca_xy) / sigma_dca_xy))

            # Depending on the chosen signal generate labels accordingly
            if self.signal == 'hf_vs_l':
                if jets[b'mTag'][i] - 1 in [1, 2]:
                    self.X.append(jet_matrix)
                    self.y.append(1)
                    self.jets += 1
                else:
                    self.X.append(jet_matrix)
                    self.y.append(0)
                    self.jets += 1
            elif self.signal == 'c_vs_b':
                if jets[b'mTag'][i] - 1 == 2:
                    self.X.append(jet_matrix)
                    self.y.append(1)
                    self.jets += 1
                elif jets[b'mTag'][i] - 1 == 1:
                    self.X.append(jet_matrix)
                    self.y.append(0)
                    self.jets += 1
            elif self.signal == 'c_vs_all':
                if jets[b'mTag'][i] - 1 == 1:
                    self.X.append(jet_matrix)
                    self.y.append(1)
                    self.jets += 1
                else:
                    self.X.append(jet_matrix)
                    self.y.append(0)
                    self.jets += 1
            elif self.signal == 'b_vs_all':
                if jets[b'mTag'][i] - 1 == 2:
                    self.X.append(jet_matrix)
                    self.y.append(1)
                    self.jets += 1
                else:
                    self.X.append(jet_matrix)
                    self.y.append(0)
                    self.jets += 1

    def __len__(self):
        return self.jets

    def __getitem__(self, idx):
        X = self.X[idx].copy()
        if self.track_shuffling:
            np.random.shuffle(X)
        return torch.Tensor(X), torch.IntTensor([self.y[idx]])

    def _calculate_normalization(self, variable, jets):
        var = list()

        if variable != 'fM':
            for i in range(len(self.file)):
                var.extend(jets[bytes(variable, encoding='utf-8')][i])
        else:
            for i in range(len(self.file)):
                var.extend(jets[b'fZ'][i]*jets[b'fDeltaR'][i]*jets[b'fDeltaR'][i])

        mu_var, sigma_var = np.mean(var), np.std(var)

        del var

        return mu_var, sigma_var

# To speedup training take the jet with the largest number of tracks and pad other jets with zeros
# to match size of the largest jet
def collate_fn(batch):
    last_dim = list()

    for item in batch:
        last_dim.append(item[0].shape[0])
    largest_dim = np.max(last_dim)

    data = list()
    for item in batch:
        if item[0].shape[0] < largest_dim:
            pad = torch.Tensor(largest_dim - item[0].shape[0], item[0].shape[1])
            pad.fill_(0)
            data.append(torch.cat((item[0], pad), dim=0))
        else:
            data.append(item[0])
    target = [item[1] for item in batch]
    return torch.stack(data), torch.LongTensor(target)

# The following was taken and modified from the https://github.com/pytorch/examples/blob/master/imagenet/main.py
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, acc, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (X, y) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        X = X.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        # compute output
        output = model(X)
        loss = criterion(output, y)

        # measure accuracy and record loss
        acc1 = accuracy(output, y, topk=(1, ))
        losses.update(loss.item(), X.size(0))
        acc.update(acc1[0].item(), X.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step((i+1)/len(train_loader))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            progress.print(i)

    return losses.avg


def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, acc, prefix='Val: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (X, y) in enumerate(val_loader):

            X = X.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            # compute output
            output = model(X)
            loss = criterion(output, y)

            # measure accuracy and record loss
            acc1 = accuracy(output, y, topk=(1, ))
            losses.update(loss.item(), X.size(0))
            acc.update(acc1[0].item(), X.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                progress.print(i)

        print(' * Val loss {losses.avg:.3f}'.format(losses=losses))

    return losses.avg, acc.avg


def test(test_loader, log_path, model, generate_curve=False):
    y_pred = list()
    y_true = list()
    prob = torch.nn.Softmax(dim=1)

    model.eval()

    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):

            X = X.cuda(non_blocking=True)

            output = model(X)
            y_pred.extend(prob(output.cpu()).numpy()[:, 1])
            y_true.extend(y.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    pr, rc, _ = precision_recall_curve(y_true, y_pred)

    if generate_curve:
        np.savetxt((log_path.with_name(log_path.with_suffix('').name + '_roc.txt')).as_posix(), (fpr, tpr))
        np.savetxt((log_path.with_name(log_path.with_suffix('').name + '_pr.txt')).as_posix(), (pr, rc))

    return auc(fpr, tpr)
