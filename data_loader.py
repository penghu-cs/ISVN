import scipy.io as sio
import h5py
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
from torchvision import datasets, transforms


class NDataset(Dataset):
    def __init__(self, data, labels=None, transform=None, is_path=False, root='./'):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.is_path = is_path
        self.root = root

    def __getitem__(self, index):
        return self.data[index].astype('float32') if self.transform is None else self.transform(self.data[index]).float(), self.labels[index] if self.labels is not None else -1

    def __len__(self):
        return len(self.labels)


def load_data(data_name, K=400, view=0):
    if (type(data_name) is list) and (view == -1):
        ret = [[], [], [], [], [], [], [], [], [], []]
        for dn in data_name:
            data = load_deep_features(dn, K=K, view=view)
            ret = [r + [d] for r, d in zip(ret, data[0: -1])]
        return ret + [data[-1]]
    elif view == -1:
        ret = [[], [], [], [], [], [], [], [], [], []]
        for v in range(2):
            data = load_deep_features(data_name, K=K, view=v)
            ret = [r + [d] for r, d in zip(ret, data[0: -1])]
        return ret + [data[-1]]
    else:
        data_name = data_name[view] if type(data_name) is list else data_name
        dd = load_deep_features(data_name, K=K, view=view)
        return [[d] for d in dd[0: -1]] + [dd[-1]]
    

def load_deep_features(data_name, K=400, view=0):
    train_transform, test_transform = None, None

    if data_name == 'xmedianet':
        valid_len, MAP = 4000, 0
        split = 'img' if view == 0 else 'text'
        path = './data/XMediaNet/xmedianet_deep_doc2vec_data.h5py'
        with h5py.File(path, 'r') as h:
            data, labels = h['train_%s' % (split + 's_deep' if view == 0 else split)][()].astype('float32'), h['train_%ss_labels' % split][()].reshape([-1])
            valid_data, valid_labels = data[-valid_len::], labels[-valid_len::]
            train_data, train_labels = data[0: -valid_len], labels[0: -valid_len]
            test_data, test_labels = h['test_%s' % (split + 's_deep' if view == 0 else split)][()].astype('float32'), h['test_%ss_labels' % split][()].reshape([-1])
    
    elif data_name == 'nus_wide':
        valid_len, MAP = 5000, 0
        path = './data/NUS-WIDE/nus_wide_deep_doc2vec_data_42941.h5py'
        split = 'img' if view == 0 else 'text'
        with h5py.File(path, 'r') as h:
            data, labels = h['train_%s' % (split + 's_deep' if view == 0 else split)][()].astype('float32'), h['train_%ss_labels' % split][()].reshape([-1])
            valid_data, valid_labels = data[-valid_len::], labels[-valid_len::]
            train_data, train_labels = data[0: -valid_len], labels[0: -valid_len]
            test_data, test_labels = h['test_%s' % (split + 's_deep' if view == 0 else split)][()].astype('float32'), h['test_%ss_labels' % split][()].reshape([-1])
        
    elif data_name == 'INRIA-Websearch':
        MAP = 0
        split = 'img' if view == 0 else 'txt'
        data = sio.loadmat('./data/INRIA-Websearch/INRIA-Websearch.mat')
        train_data, train_labels = data['tr_%s' % split].astype('float32'), data['tr_%s_lab' % split].reshape([-1])
        valid_data, valid_labels = data['val_%s' % split].astype('float32'), data['val_%s_lab' % split].reshape([-1])
        test_data, test_labels = data['te_%s' % split].astype('float32'), data['te_%s_lab' % split].reshape([-1])

    elif data_name.lower() == 'mnist':
        MAP = None
        train_dataset = datasets.MNIST('./data/MNIST/', train=True, download=True)
        data, labels = train_dataset.data.numpy(), train_dataset.train_labels.numpy()
        valid_data, valid_labels = data[-10000::], labels[-10000::]
        train_data, train_labels = data[0: -10000], labels[0: -10000]

        test_dataset = datasets.MNIST('./data/MNIST/', train=False, download=True)
        test_data, test_labels = test_dataset.data.numpy(), test_dataset.test_labels.numpy()

        train_transform = transforms.Compose([
            lambda x: x.unsqueeze(-1),
            transforms.ToPILImage(),
            # transforms.RandomResizedCrop(32),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        test_transform = transforms.Compose([
            lambda x: x.unsqueeze(-1),
            transforms.ToPILImage(),
            # transforms.Resize(32),
            # transforms.CenterCrop(32),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    elif data_name.lower() == 'svhn':
        MAP = None
        train_dataset = datasets.SVHN('./data/SVHN/', split='train', download=True)
        data, labels = train_dataset.data, train_dataset.labels
        valid_data, valid_labels = data[-10000::], labels[-10000::]
        train_data, train_labels = data[0: -10000], labels[0: -10000]

        test_dataset = datasets.SVHN('./data/SVHN/', split='test', download=True)
        test_data, test_labels = test_dataset.data, test_dataset.labels

        # inx = np.arange(train_data.shape[0])
        # np.random.shuffle(inx)
        # train_labeled_data, train_labeled_labels, train_unlabeled_data, train_unlabeled_labels = train_data[inx[0: K]], train_labels[inx[0: K]], train_data[inx[K::]], train_labels[inx[K::]]
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomResizedCrop(32),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize(32),
            # transforms.CenterCrop(32),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    inx = np.arange(train_data.shape[0])
    np.random.shuffle(inx)
    train_labeled_data, train_labeled_labels, train_unlabeled_data, train_unlabeled_labels = train_data[inx[0: K]], train_labels[inx[0: K]], train_data[inx[K::]], train_labels[inx[K::]]
    return train_labeled_data, train_labeled_labels, train_unlabeled_data, train_unlabeled_labels, valid_data, valid_labels, test_data, test_labels, train_transform, test_transform, MAP
