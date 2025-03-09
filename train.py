import os
import random
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.fcbformer.models import FCBFormer
from trainer import trainer_coca

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='COCA', help='dataset name')
parser.add_argument('--root_path', type=str, default='/home/psw/AVS-Diagnosis/COCA/COCA_1frame/train_npz', help='root dir for data')
parser.add_argument('--list_dir', type=str, default='/home/psw/AVS-Diagnosis/COCA/COCA_1frame/lists_COCA', help='list dir')
parser.add_argument('--num_classes', type=int, default=5, help='output channel of network')
parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.00001, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
parser.add_argument('--encoder', type=str, default='resnet50', help='for segmentation_models_pytorch encoder')
parser.add_argument('--exp_setting', type=str,  default='default', help='description of experiment setting')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=42, help='random seed')
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

    net = FCBFormer().cuda()
    args.encoder = 'pvt_v2_b3'
    
    """ from torchinfo import summary
    torchinfo_summary = str(summary(net, input_size=(args.batch_size, 1, args.img_size, args.img_size), 
                                    col_width=20, depth=5, 
                                    row_settings=["depth", "var_names"], 
                                    col_names=["input_size", "kernel_size", "output_size", "params_percent"]))
    output_file = "model_summary.txt"
    with open(output_file, "w") as file:
        file.write(torchinfo_summary) """

    exp_path = os.path.join(net.__class__.__name__ + '_' + args.encoder, args.dataset + '_' + str(args.img_size), args.exp_setting)
    parameter_path = 'epo' + str(args.max_epochs) + '_bs' + str(args.batch_size) + '_lr' + str(args.base_lr)
    
    snapshot_path = os.path.join("./model/", exp_path, parameter_path)

    os.makedirs(snapshot_path, exist_ok=True)

    trainer = {'COCA': trainer_coca}
    trainer[args.dataset](args, net, snapshot_path)