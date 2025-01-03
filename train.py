import os
import random
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import segmentation_models_pytorch as smp
from trainer import trainer_coca

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/COCA/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='COCA', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_COCA', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_epochs', type=int,
                    default=500, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    dataset_name = args.dataset
    dataset_config = {
        'COCA': {
            'root_path': './data/COCA_3frames/train_npz',
            'list_dir': './data/COCA_3frames/lists_COCA',
            'num_classes': 4,
            'max_epochs': 300,
            'batch_size': 16,
            'base_lr': 0.00001,
            'img_size': 512,
            'encoder': 'resnet50_sa',
            'exp_setting': 'default',
        },
    }
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.max_epochs = dataset_config[dataset_name]['max_epochs']
    args.batch_size = dataset_config[dataset_name]['batch_size']
    args.base_lr = dataset_config[dataset_name]['base_lr']
    args.img_size = dataset_config[dataset_name]['img_size']
    args.encoder = dataset_config[dataset_name]['encoder']
    args.exp_setting = dataset_config[dataset_name]['exp_setting']

    net = smp.Unet(
            encoder_name=args.encoder,
            encoder_weights="imagenet",
            in_channels=1,
            classes=args.num_classes,
            ).cuda()
    
    """ from torchinfo import summary
    torchinfo_summary = str(summary(net, 
                                    input_size=(args.batch_size, 1, args.img_size, args.img_size), 
                                    col_width=20, 
                                    depth=5, 
                                    row_settings=["depth", "var_names"], 
                                    col_names=["input_size", "kernel_size", "output_size", "params_percent"]))
    output_file = "model_summary.txt"
    with open(output_file, "w") as file:
        file.write(torchinfo_summary) """

    snapshot_path = f"./model/{net.__class__.__name__ + '_' + args.encoder}/{dataset_name + '_' + str(args.img_size)}/{args.exp_setting}/{'epo' + str(args.max_epochs)}"
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        
    trainer = {'COCA': trainer_coca}
    trainer[dataset_name](args, net, snapshot_path)