import torch
import numpy as np
import sys
import scipy.spatial
import scipy.io as sio
import os
from sklearn.neighbors import KNeighborsClassifier
import scipy

def getOrthW(num_classes, output_shape):
    file_name = 'Orth_Ws/Orth_W_C%d_O%d.mat' % (num_classes, output_shape)
    W = torch.Tensor(output_shape, output_shape)
    W = torch.nn.init.orthogonal(W, gain=1)[:, 0: num_classes].numpy()
    if os.path.exists(file_name):
        W = sio.loadmat(file_name)['W']
    else:
        sio.savemat(file_name, {'W': W})
    return W

def save_checkpoint(state, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    path = os.path.join(prefix, filename)
    while tries:
        try:
            torch.save(state, path)
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error

def to_tensor(x, cuda_id=0):
    x = torch.tensor(x)
    if torch.cuda.is_available():
        x = x.cuda(cuda_id)
    return x

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.numpy()

def multi_test(data, data_labels, MAP=None, metric='cosine'):
    n_view = len(data)
    res = np.zeros([n_view, n_view])
    if MAP is None:
        for i in range(n_view):
            for j in range(n_view):
                if i == j:
                    continue
                else:
                    neigh = KNeighborsClassifier(n_neighbors=1, metric=metric)
                    neigh.fit(data[j], data_labels[j])
                    la = neigh.predict(data[i])
                    res[i, j] = np.sum((la == data_labels[i].reshape([-1])).astype(int)) / float(la.shape[0])
    else:
        if MAP == -1:
            res = [np.zeros([n_view, n_view]), np.zeros([n_view, n_view])]
        for i in range(n_view):
            for j in range(n_view):
                if i == j:
                    continue
                else:
                    if len(data_labels[j].shape) == 1:
                        tmp = fx_calc_map_label(data[j], data_labels[j], data[i], data_labels[i], MAP, metric=metric)
                    else:
                        Ks = [50, 0] if MAP == -1 else [MAP]
                        tmp = []
                        for k in Ks:
                            tmp.append(fx_calc_map_multilabel_k(data[j], data_labels[j], data[i], data_labels[i], k=k, metric=metric))
                    if MAP == -1:
                        for _i in range(len(tmp)):
                            res[_i][i, j] = tmp[_i]
                    else:
                        res[i, j] = tmp[0]
    return res

def fx_calc_map_label(train, train_labels, test, test_label, k=0, metric='cosine'):
    dist = scipy.spatial.distance.cdist(test, train, metric)
    ord = dist.argsort(1)
    # numcases = dist.shape[1]
    numcases = train_labels.shape[0]
    if k == 0:
        k = numcases
    if k == -1:
        ks = [50, numcases]
    else:
        ks = [k]

    def calMAP(_k):
        _res = []
        for i in range(len(test_label)):
            order = ord[i]
            p = 0.0
            r = 0.0
            for j in range(_k):
                if test_label[i] == train_labels[order[j]]:
                    r += 1
                    p += (r / (j + 1))
            if r > 0:
                _res += [p / r]
            else:
                _res += [0]
        return np.mean(_res)

    res = []
    for k in ks:
        res.append(calMAP(k))
    return res

def predict(model, dataloader, device=0):
    results, labels = [], []
    with torch.no_grad():
        for _, (d, t) in enumerate(dataloader):
            batch = to_tensor(d, device)
            results.append(to_data(model(batch)))
            labels.append(t)
    return np.concatenate(results), np.concatenate(labels)

def show_progressbar(rate, *args, **kwargs):
    '''
    :param rate: [current, total]
    :param args: other show
    '''
    inx = rate[0] + 1
    count = rate[1]
    bar_length = 30
    rate[0] = int(np.around(rate[0] * float(bar_length) / rate[1])) if rate[1] > bar_length else rate[0]
    rate[1] = bar_length if rate[1] > bar_length else rate[1]
    num = len(str(count))
    str_show = ('\r%' + str(num) + 'd / ' + '%' + str(num) + 'd  (%' + '3.2f%%) [') % (inx, count, float(inx) / count * 100)
    for i in range(rate[0]):
        str_show += '='

    if rate[0] < rate[1] - 1:
        str_show += '>'

    for i in range(rate[0], rate[1] - 1, 1):
        str_show += '.'
    str_show += '] '
    for l in args:
        str_show += ' ' + str(l)

    for key in kwargs:
        try:
            str_show += ' ' + key + ': %.4f' % kwargs[key]
        except Exception:
            str_show += ' ' + key + ': ' + str(kwargs[key])
    if inx == count:
        str_show += '\n'

    sys.stdout.write(str_show)
    sys.stdout.flush()
