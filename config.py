import argparse

def parse_args():
    desc ="Pytorch CAP-NET"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--isTrain', action='store_true', help='[train/test]')
    parser.add_argument('--dataset', type=str, required=True, help='[mnist/cifar10/imagenet]')
    parser.add_argument('--datapath', type=str, required=True, help='Denote the dataset path')
    parser.add_argument('--img_size', type=int, default=32, help='[mnist:28 / cifar:32 / imagenet:256]')
    parser.add_argument('--channel', type=int, default=3, help='[mnist:1 / cifar:3 / imagenet:3]')
    parser.add_argument('--ngf', type=int, default=64, help='[mnist:32 / cifar:64 / imagenet:64]')
    parser.add_argument('--ndf', type=int, default=64, help='[mnist:32 / cifar:64 / imagenet:64]')
    parser.add_argument('--epoch', type=int, default=201, help='The number of iteration to train the model')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for PGAN')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--save_freq', type=int, default=10, help='The number of saving models')
    parser.add_argument('--result_dir', type=str, default='./samples/')
    parser.add_argument('--model', type=str, default='res_simple')
    parser.add_argument('--num_step', type=int, default='40')

    parser.add_argument('--whx', action='store_true', help='[whitebox or blackbox]')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoint/')
    parser.add_argument('--name', type=str, default='results', help='name of result directory')
    parser.add_argument('--attack', type=str, default='FGSM', help='The type of adversarial attacks')
    parser.add_argument('--epsilons', type=int, default=8, help='The magnitude of the adversarial attacks')
    parser.add_argument('--load_epoch', type=int, default=0, help='The number of epoch to test the model')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()
