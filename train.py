import os
import random
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import segmentation_models_pytorch as smp
from trainer import trainer_coca

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='COCA', help='dataset name')
parser.add_argument('--root_path', type=str, default='/home/psw/AVS-Diagnosis/COCA/COCA_3frames/train_npz', help='root dir for data')
parser.add_argument('--list_dir', type=str, default='/home/psw/AVS-Diagnosis/COCA/COCA_3frames/lists_COCA', help='list dir')
parser.add_argument('--num_classes', type=int, default=5, help='output channel of network')
parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.00001, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
parser.add_argument('--encoder', type=str, default='resnet50_sa', help='for segmentation_models_pytorch encoder', choices=['resnet50_sa', 'densenet201_sa', 'efficientnet-b4_sa', 'mit_b2_sa'])
parser.add_argument('--decoder', type=str, default='unet', help='for segmentation_models_pytorch decoder', choices=['unet', 'segformer'])
parser.add_argument('--exp_setting', type=str,  default='default', help='description of experiment setting')
parser.add_argument('--finetune_exp_setting', type=str, default='', help='description of experiment setting for finetuning')
parser.add_argument('--enable_finetuning', action="store_true", help='Path to model checkpoint for finetuning')
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

    if args.decoder == 'unet':
        net = smp.Unet(encoder_name=args.encoder,
                       encoder_weights="imagenet",
                       in_channels=1,
                       classes=args.num_classes).cuda()
    elif args.decoder == 'segformer':
        net = smp.Segformer(encoder_name=args.encoder,
                            encoder_weights="imagenet",
                            in_channels=1,
                            classes=args.num_classes).cuda()

    """ from torchinfo import summary
    torchinfo_summary = str(summary(net, input_size=(args.batch_size, 1, args.img_size, args.img_size), 
                                    col_width=20, depth=5, 
                                    row_settings=["depth", "var_names"], 
                                    col_names=["input_size", "kernel_size", "output_size", "params_percent"]))
    output_file = f"{net.__class__.__name__}_{args.encoder}_model_summary.txt"
    with open(output_file, "w") as file:
        file.write(torchinfo_summary) """

    exp_path = os.path.join(net.__class__.__name__ + '_' + args.encoder, args.dataset + '_' + str(args.img_size), args.exp_setting)
    parameter_path = 'epo' + str(args.max_epochs) + '_bs' + str(args.batch_size) + '_lr' + str(args.base_lr)
    snapshot_path = os.path.join("./model/", exp_path, parameter_path)
    os.makedirs(snapshot_path, exist_ok=True)
    
    # Finetuning: load pretrained checkpoint if provided
    if args.enable_finetuning:
        files = os.listdir(snapshot_path)
        best_model_file = None
        for f in files:
            name, ext = os.path.splitext(f)
            if name.endswith("best_model"):
                best_model_file = f
                break
        if best_model_file is None:
            raise FileNotFoundError("No checkpoint ending with 'best_model' found in " + snapshot_path)
        
        checkpoint_path = os.path.join(snapshot_path, best_model_file)
        checkpoint = torch.load(checkpoint_path)
        # Remove segmentation head weights to avoid size mismatch (checkpoint was trained for 5 classes)
        for key in list(checkpoint.keys()):
            if key.startswith("segmentation_head."):
                del checkpoint[key]
        net.load_state_dict(checkpoint, strict=False)
        print(f"Loaded checkpoint from {best_model_file}")
        
        finetune_exp_path = os.path.join(net.__class__.__name__ + '_' + args.encoder, args.dataset + '_' + str(args.img_size), args.finetune_exp_setting)
        finetune_snapshot_path = os.path.join("./model/", finetune_exp_path, parameter_path)
        snapshot_path = finetune_snapshot_path
        os.makedirs(snapshot_path, exist_ok=True)
        
    trainer = {'COCA': trainer_coca}
    trainer[args.dataset](args, net, snapshot_path)