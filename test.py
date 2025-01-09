import os
import sys
import random
import argparse
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import segmentation_models_pytorch as smp
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.dataset import COCA_dataset
from utils import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str, default='./data/COCA/test_vol_h5', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str, default='COCA', help='experiment_name')
parser.add_argument('--list_dir', type=str, default='./data/COCA/lists_COCA', help='list dir')
parser.add_argument('--num_classes', type=int, default=5, help='output channel of network')
parser.add_argument('--max_epochs', type=int, default=1000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=96, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--test_save_dir', type=str, default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
args = parser.parse_args()

def inference(args, model, test_save_path=None):
    db_test = COCA_dataset(
        base_dir=args.volume_path, 
        split="test", 
        list_dir=args.list_dir,
    )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    
    metric_list_all = []
    
    for i_batch, sampled_batch in tqdm(enumerate(testloader, start=1)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list_all.append(metric_i)
        
        mean_dice_case = np.nanmean(metric_i, axis=0)[0]
        mean_m_ap_case = np.nanmean(metric_i, axis=0)[1]
        mean_hd95_case = np.nanmean(metric_i, axis=0)[2]
        logging.info('%s - mean_dice: %.4f, mean_m_ap: %.4f, mean_hd95: %.2f' % (case_name, mean_dice_case, mean_m_ap_case, mean_hd95_case))
    
    metric_array = np.array(metric_list_all)
    
    for i in range(1, args.num_classes):
        class_dice = np.nanmean(metric_array[:, i-1, 0])
        class_m_ap = np.nanmean(metric_array[:, i-1, 1])
        class_hd95 = np.nanmean(metric_array[:, i-1, 2])
        logging.info('Mean class %d - mean_dice: %.4f, mean_m_ap: %.4f, mean_hd95: %.2f' % (i, class_dice, class_m_ap, class_hd95))
        
    mean_dice = np.nanmean(metric_array[:,:,0])
    mean_m_ap = np.nanmean(metric_array[:,:,1])
    mean_hd95 = np.nanmean(metric_array[:,:,2])
    
    logging.info('Testing performance in best val model - mean_dice : %.4f, mean_m_ap : %.4f, mean_hd95 : %.2f' % (mean_dice, mean_m_ap, mean_hd95))
    
    return "Testing Finished!"

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
            'Dataset': COCA_dataset,
            'volume_path': '/home/psw/AVS-Diagnosis/COCA_1frame/test_vol_h5',
            'list_dir': '/home/psw/AVS-Diagnosis/COCA_1frame/lists_COCA',
            'num_classes': 5,
            'max_epochs': 300,
            'batch_size': 16,
            'base_lr': 0.00001,
            'img_size': 512,
            'encoder': 'resnet50',
            'exp_setting': 'default',
            'z_spacing': 3,
        },
    }
    
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.max_epochs = dataset_config[dataset_name]['max_epochs']
    args.batch_size = dataset_config[dataset_name]['batch_size']
    args.base_lr = dataset_config[dataset_name]['base_lr']
    args.img_size = dataset_config[dataset_name]['img_size']
    args.encoder = dataset_config[dataset_name]['encoder']
    args.exp_setting = dataset_config[dataset_name]['exp_setting']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    
    net = smp.Unet(
            encoder_name=args.encoder,            # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",         # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1,                      # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=args.num_classes,           # model output channels (number of classes in your dataset)
            ).cuda()
    
    exp_path = os.path.join(net.__class__.__name__ + '_' + args.encoder, dataset_name + '_' + str(args.img_size), args.exp_setting)
    parameter_path = 'epo' + str(args.max_epochs) + '_bs' + str(args.batch_size) + '_lr' + str(args.base_lr)
    
    snapshot_path = os.path.join("./model/", exp_path, parameter_path)
    best_model_path = glob(os.path.join(snapshot_path, '*_best_model.pth'))[0] # 유일한 best_model 파일 선택
    if not best_model_path:
        raise FileNotFoundError(f"Best model not found at {snapshot_path}")
    net.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model from: {best_model_path}")
    
    log_path = os.path.join("./test_log", exp_path, parameter_path)
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(filename=log_path + "/" + "results.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(best_model_path)
    logging.info(str(args))
    logging.info(parameter_path)

    if args.is_savenii:
        test_save_path = os.path.join(log_path, "results_nii")
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    inference(args, net, test_save_path)