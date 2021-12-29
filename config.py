import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=list, default=[2e-4, 2e-4, 2e-4, 2e-4, 2e-4])
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--checkpoint', type=str, default='checkpoint/')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--lr_update', type=int, default=200)
parser.add_argument('--gpu_id', type=int, default=0)

parser.add_argument('--batch_size', type=int, default=16) ## 64
parser.add_argument('--output_shape', type=int, default=1024)
parser.add_argument('--alpha', type=float, default=0.)
parser.add_argument('--threshold', type=float, default=0.7)  # MNIST_SVHN: 0.9 INRIA-Websearch 0.7
parser.add_argument('--beta', type=float, default=0)  # 400: 8e-2 1000: 3e-2 2000: 2e-2 3000: 2e-2 4000: 2e-2
parser.add_argument('--view_id', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--K', type=int, default=2000) # 400 1000 2000 3000 4000
parser.add_argument('--CNN', action='store_true', default=False)
parser.add_argument('--multiprocessing', action='store_true', default=False)
parser.add_argument('--datasets', nargs='+', help='<Required> Quantization bits', default='INRIA-Websearch') # nus_wide xmedianet INRIA-Websearch MNIST SVHN

args = parser.parse_args()
if len(args.datasets) == 1:
    args.datasets = args.datasets[0]
