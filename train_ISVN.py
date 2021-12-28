import torch
import time
import utils
import data_loader
import scipy.io as sio
import torch.utils.data as data
from ISVN import ISVN

class Solver(object):
    def __init__(self, config):
        wv_matrix = None
        self.args = config
        self.output_shape = config.output_shape
        self.multiprocessing = config.multiprocessing
        self.datasets = config.datasets
        self.view = config.view_id
        self.K = config.K
        (self.train_labeled_data, self.train_labeled_labels, self.train_unlabeled_data, self.train_unlabeled_labels, self.valid_data, self.valid_labels, self.test_data, self.test_labels, train_transforms, test_transforms, self.MAP) = data_loader.load_data(self.datasets, self.K, self.view)
        self.n_view = len(self.train_labeled_data)# if type(self.train_labeled_data) is list else 1

        if not config.CNN:
            self.train_labeled_data = [self.train_labeled_data[i].reshape([self.train_labeled_data[i].shape[0], -1]) for i in range(self.n_view)]
            self.train_unlabeled_data = [self.train_unlabeled_data[i].reshape([self.train_unlabeled_data[i].shape[0], -1]) for i in range(self.n_view)]
            self.valid_data = [self.valid_data[i].reshape([self.valid_data[i].shape[0], -1]) for i in range(self.n_view)]
            self.test_data = [self.test_data[i].reshape([self.test_data[i].shape[0], -1]) for i in range(self.n_view)]

            max_value = [self.train_labeled_data[i].max() for i in range(self.n_view)]
            self.train_labeled_data = [self.train_labeled_data[i] / max_value[i] for i in range(self.n_view)]
            self.train_unlabeled_data = [self.train_unlabeled_data[i] / max_value[i] for i in range(self.n_view)]
            self.valid_data = [self.valid_data[i] / max_value[i] for i in range(self.n_view)]
            self.test_data = [self.test_data[i] / max_value[i] for i in range(self.n_view)]

            train_transforms, test_transforms = [None for i in range(len(train_transforms))], [None for i in range(len(test_transforms))]

        num_workers = self.args.num_workers
        self.models, self.train_labeled_dataloaders, self.train_unlabeled_dataloaders, self.valid_dataloaders, self.test_dataloaders = [], [], [], [], []
        for v in range(self.n_view):
            view = v if self.view < 0 else self.view
            if config.mode == 'train':
                train_labeled_dataset = data_loader.NDataset(self.train_labeled_data[v], self.train_labeled_labels[v], transform=train_transforms[v])
                self.train_labeled_dataloaders.append(data.DataLoader(train_labeled_dataset, batch_size=config.batch_size, shuffle=True, num_workers=num_workers, drop_last=False))

                train_unlabeled_dataset = data_loader.NDataset(self.train_unlabeled_data[v], self.train_unlabeled_labels[v], transform=train_transforms[v])
                self.train_unlabeled_dataloaders.append(data.DataLoader(train_unlabeled_dataset, batch_size=config.batch_size, shuffle=True, num_workers=num_workers, drop_last=False))

                valid_dataset = data_loader.NDataset(self.valid_data[v], self.valid_labels[v], transform=test_transforms[v])
                self.valid_dataloaders.append(data.DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers, drop_last=False))
                self.models.append(ISVN(config, self.train_labeled_dataloaders[v], self.train_unlabeled_dataloaders[v], self.valid_dataloaders[v], view))
            else:
                test_dataset = data_loader.NDataset(self.test_data[v], self.test_labels[v], transform=test_transforms[v])
                self.test_dataloaders.append(data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers, drop_last=False))
                self.models.append(ISVN(config, self.test_dataloaders[v], None, None, view))

    def getDevice(self, v):
        return self.args.gpu_id if self.args.gpu_id >= 0 else (v + 1) % torch.cuda.device_count()

    def train(self):
        start = time.time()
        if self.multiprocessing and self.view < 0:
            # Old PyTorch Version <= 1.1.0
            # import torch.multiprocessing as mp
            # mp = mp.get_context('spawn')
            # self.resutls = mp.Manager().list(self.resutls)
            # process = []
            # start = time.time()
            # for v in range(self.n_view):
            #     process.append(mp.Process(target=self.train_view, args=(v,)))
            #     process[v].daemon = True
            # for v in range(self.n_view):
            #     process[v].start()
            # start = time.time()
            # for p in process:
            #     p.join()

            # New PyTorch Version >= 1.2.0
            import threading
            ths = []
            for v in range(self.n_view):
                cuda_id = self.getDevice(v)
                ths.append(threading.Thread(target=self.models[v].train_view, args=(cuda_id,)))
            for v in range(self.n_view):
                ths[v].start()
            for p in ths:
                p.join()

        elif self.view < 0:
            start = time.time()
            for v in range(self.n_view):
                cuda_id = self.getDevice(v)
                self.models[v].train_view(cuda_id)
        else:
            start = time.time()
            cuda_id = self.getDevice(self.view)
            self.models[0].train_view(cuda_id)
        end = time.time()
        runing_time = end - start
        print('The training time: ' + str(runing_time))

    def eval(self):
        for v in range(self.n_view):
            self.models[v].load_checkpoint()
            # print('View #%d: %d' % (v, self.retrieval_dataloader[v].dataset.data.shape[1]))

        test = [self.models[v].eval(self.test_dataloaders[v], self.getDevice(0)) for v in range(self.n_view)]
        test_results = utils.multi_test([test[v][0] for v in range(self.n_view)], [test[v][1] for v in range(self.n_view)], self.MAP)
        print("test results: " + self.view_result(test_results))
        sio.savemat('features/%s_O%d_K%d_B%g_A%g_T%g.mat' % (self.args.datasets, self.output_shape, self.args.K, self.args.beta, self.args.alpha, self.args.threshold), {'test': [test[v][0] for v in range(self.n_view)], 'test_labels': [test[v][1] for v in range(self.n_view)]})
        return test_results

    def view_result(self, _acc):
        res = ''
        if type(_acc) is not list:
            res += ((' - mean: %.3f' % (np.sum(_acc) / (self.n_view * (self.n_view - 1)))) + ' - detail: ')
            for _i in range(self.n_view):
                for _j in range(self.n_view):
                    if _i != _j:
                        res += ('%.3f' % _acc[_i, _j]) + ' , '
        else:
            R = [50, 'ALL']
            for _k in range(len(_acc)):
                res += (' R = ' + str(R[_k]) + ': ')
                res += ((' - mean: %.3f' % (np.sum(_acc[_k]) / (self.n_view * (self.n_view - 1)))) + ' - detail: ')
                for _i in range(self.n_view):
                    for _j in range(self.n_view):
                        if _i != _j:
                            res += ('%.3f' % _acc[_k][_i, _j]) + ' , '
        return res


def main(args):
    solver = Solver(args)
    cudnn.benchmark = True
    if args.mode == 'train':
        ret = solver.train()
    elif args.mode == 'eval':
        ret =  solver.eval()
    print(args)
    return ret

if __name__ == '__main__':
    from config import args
    import numpy as np
    from torch.backends import cudnn
    cudnn.enabled = False
    results = main(args)