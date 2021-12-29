import torch
import numpy as np
import os
from torch import optim
import utils
from torch.autograd import Variable
from model import Dense_Net, ConvEncoder, ConvDecoder, Dense_Net
import copy


class ISVN(object):
    def __init__(self, config, train_labeled_dataloader, train_unlabeled_dataloader, valid_dataloader, view, W, classes):
        self.args = config
        self.output_shape = config.output_shape
        self.seed = config.seed
        self.threshold = config.threshold
        self.train_labeled_dataloader = train_labeled_dataloader
        self.train_unlabeled_dataloader = train_unlabeled_dataloader
        self.valid_dataloader = valid_dataloader

        self.view = view
        self.input_shape = self.train_labeled_dataloader.dataset.data.shape[1]
        self.classes = classes
        

        if len(self.train_labeled_dataloader.dataset.data.shape) > 2:
            c_in = self.train_labeled_dataloader.dataset.data.shape[-1] if len(self.train_labeled_dataloader.dataset.data.shape) == 4 else 1
            self.Encoder = ConvEncoder(c_in, self.output_shape, bn=False, norm=True)
            self.Decoder = ConvDecoder(self.output_shape, c_in, bn=False, norm=False)
            self.isNorm = False
        else:
            self.Encoder = Dense_Net(input_dim=self.input_shape, out_dim=self.output_shape, norm=True)
            self.Decoder = Dense_Net(input_dim=self.output_shape, out_dim=self.input_shape, norm=True)
            self.isNorm = True

        self.lr = config.lr[view]
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.batch_sizes = config.batch_size

        self.epochs = config.epochs
        self.alpha = config.alpha
        self.beta = config.beta
        if isinstance(self.args.datasets, list):
            self.checkpoint_file = '{}_checkpoint_O{}_K{}_B{}_A{}_T{}.pth.tar'.format(self.args.datasets[self.view], self.output_shape, self.args.K, self.alpha, self.beta, self.threshold)
        else:
            self.checkpoint_file = '{}_checkpoint_V{}_O{}_K{}_B{}_A{}_T{}.pth.tar'.format(self.args.datasets, self.view, self.output_shape, self.args.K, self.alpha, self.beta, self.threshold)
        self.W = W

    def to_var(self, x, cuda_id):
        """Converts numpy to variable."""
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        if torch.cuda.is_available():
            x = x.cuda(cuda_id)
        return Variable(x)  # torch.autograd.Variable

    def to_data(self, x):
        """Converts variable to numpy."""
        try:
            if torch.cuda.is_available():
                x = x.cpu()
            return x.data.numpy()
        except Exception as e:
            return x

    def to_one_hot(self, x, device=0):
        if len(x.shape) == 1 or x.shape[1] == 1:
            if isinstance(x, torch.Tensor):
                one_hot = (self.to_var(self.classes, device).view([1, -1]).long() == x.view([-1, 1]).long()).float().detach()
                # zeros = np.zeros([one_hot.shape[0], self.num_classes], dtype='float32')
                # labels = np.concatenate([one_hot, zeros], 1)
            else:
                one_hot = (self.classes.reshape([1, -1]) == x.reshape([-1, 1])).astype('float32')
            labels = one_hot
            y = self.to_var(labels, device)
        else:
            y = self.to_var(x, device)
        return y

    def criterion(self, x, y, labels, W):
        l2 = lambda _x, _y: ((_x - _y) ** 2).sum(1).mean()
        if isinstance(x, torch.Tensor):
            dist = x.mm(y.t()) / 2.
            sim = (labels.float().mm(labels.float().t()) > 0).float()
            loss1 = ((1. + dist.double().exp()).log() - (sim * dist).float()).sum(1).mean().float()
            loss2 = l2(x.mm(W.t()), labels)
            return self.alpha * loss1 + (1 - self.alpha) * loss2
        else:
            return (1 - self.alpha) * l2(x, y) + self.alpha * l2(np.dot(x, self.W.T), labels)

    def train_view(self, device):
        print('Start %d-th ISVN!' % self.view)
        seed = self.seed
        import numpy as np
        np.random.seed(seed)
        import random as rn
        import torch
        rn.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if torch.cuda.is_available():
            self.Encoder.cuda(device)
            self.Decoder.cuda(device)
        W = torch.tensor(self.W, requires_grad=False).cuda(device).float()
        get_grad_params = lambda model: [x for x in model.parameters() if x.requires_grad]
        params = get_grad_params(self.Encoder) + get_grad_params(self.Decoder)
        optimizer = optim.Adam(params, self.lr, [self.beta1, self.beta2])

        discriminator_losses, losses, valid_results = [], [], []
        ae_criterion = lambda x, y: ((((x - y) ** 2).sum(1) + 1e-20).sqrt()).mean()
        label_criterion = lambda x, y: ((((x - y) ** 2).sum(1) + 1e-20).sqrt() / y.sum(1).reshape([-1, 1])).mean()
        norm = lambda x: x / torch.norm(x, dim=1, keepdim=True)

        has_unlabel = self.train_unlabeled_dataloader is not None
        batch_count = len(self.train_labeled_dataloader)
        tr_c_loss, tr_r_loss, tr_loss, val_loss = [], [], [], []
        for epoch in range(self.epochs):
            print(('\nView ID: %d, Epoch %d/%d') % (self.view, epoch + 1, self.epochs))
            labeled_train_iter = iter(self.train_labeled_dataloader)
            if has_unlabel:
                unlabeled_train_iter = iter(self.train_unlabeled_dataloader)
            mean_tr_c_loss, mean_tr_r_loss, mean_loss = [], [], []
            self.Encoder.train()
            for batch_idx in range(batch_count):
                try:
                    x_l, y_l = labeled_train_iter.next()
                    x_l, y_l = self.to_var(x_l, device), self.to_var(y_l, device)
                except Exception as e:
                    batch_count = batch_idx + 1
                    # print(e)
                    break

                if has_unlabel:
                    try:
                        x_u, y_u = unlabeled_train_iter.next()
                        x_u, y_u = self.to_var(x_u, device), self.to_var(y_u, device)
                    except Exception as e:
                        batch_count = batch_idx + 1
                        break
                    x = torch.cat([x_l, x_u])
                    y = torch.cat([y_l])
                else:
                    x = x_l
                    y = y_l

                train_y = self.to_one_hot(y, device)
                train_x = self.to_var(x, device)

                out_nets = self.Encoder(train_x)
                out_net = out_nets[-1]
                c_pred = out_net.view([out_net.shape[0], -1]).mm(W)

                r_data = norm(train_x) if self.isNorm else train_x
                r_pred = self.Decoder(out_net)[-1]

                labeled_c_loss = label_criterion(c_pred[0: x_l.shape[0]], train_y[0: x_l.shape[0]])
                labeled_r_loss = ae_criterion(r_pred[0: x_l.shape[0]], r_data[0: x_l.shape[0]])
                labeled_loss = (1 - self.alpha) * labeled_c_loss + self.alpha * labeled_r_loss

                if self.beta > 0 and has_unlabel:
                    with torch.no_grad():
                        pseudo_label = c_pred[x_l.shape[0]::]
                        pseudo_label_tmp = (pseudo_label / pseudo_label.max(1, keepdim=True)[0] > self.threshold).float()
                        pseudo_label = pseudo_label_tmp.detach()
                    unlabeled_c_loss = label_criterion(c_pred[x_l.shape[0]::], pseudo_label)
                    unlabeled_r_loss = ae_criterion(r_pred[x_l.shape[0]::], r_data[x_l.shape[0]::])
                    unlabeled_loss = (1. - self.alpha) * unlabeled_c_loss + self.alpha * unlabeled_r_loss
                    loss = labeled_loss + unlabeled_loss * self.beta
                else:
                    loss = labeled_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                mean_loss.append(loss.item())
                mean_tr_c_loss.append(labeled_c_loss.item())
                mean_tr_r_loss.append(labeled_r_loss.item())
                utils.show_progressbar([batch_idx, batch_count], loss=(loss.item() if batch_idx < batch_count - 1 else np.mean(mean_loss)))
            tr_c_loss.append(np.mean(mean_tr_c_loss))
            tr_r_loss.append(np.mean(tr_r_loss))
            tr_loss.append(np.mean(mean_loss))
            self.adjust_learning_rate(optimizer, epoch + 1)
            utils.save_checkpoint({
                'epoch': epoch,
                'model': self.Encoder.state_dict(),
                'opt': self.args,
                'loss': np.array(losses)
            }, filename=self.checkpoint_file, prefix=self.args.checkpoint)
            best_model = copy.deepcopy(self.Encoder)
        return best_model

    def eval(self, eval_dataloader, cuda_id):
        if torch.cuda.is_available():
            self.Encoder = self.Encoder.cuda(cuda_id)
        self.Encoder.eval()
        rep, lab = utils.predict(lambda x: self.Encoder(x)[-1].view([x.shape[0], -1]), eval_dataloader, device=cuda_id)
        return rep, lab

    def adjust_learning_rate(self, optimizer, epoch):
        """
        Sets the learning rate to the initial LR
        decayed by 10 after opt.lr_update epoch
        """
        if (epoch % self.args.lr_update) == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

    def load_checkpoint(self, checkpoint_file=None):
        checkpoint_file = os.path.join(self.args.checkpoint, self.checkpoint_file) if checkpoint_file is None else checkpoint_file
        ckp = torch.load(checkpoint_file)
        self.Encoder.load_state_dict(ckp['model'])
        print('Load pretrained model at %d-th epoch.' % ckp['epoch'])
        print(ckp['opt'])
        return ckp['epoch'], ckp['model'], ckp['opt'], ckp['loss']

