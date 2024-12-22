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
from torchvision import transforms as T
from datasets.dataset import COCA_dataset, ToTensor
from utils import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str, default='./data/COCA/test_npz', help='root dir for test npz data')
parser.add_argument('--dataset', type=str, default='COCA', help='experiment_name')
parser.add_argument('--list_dir', type=str, default='./data/COCA/lists_COCA', help='list dir')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
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
    test_transform = T.Compose([
        ToTensor()  # (H,W,3) -> (3,H,W)
    ])
    
    db_test = COCA_dataset(
        base_dir=args.volume_path, 
        split="test", 
        list_dir=args.list_dir,
        transform=test_transform
    )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    
    metric_list_all = []
    
    for i_batch, sampled_batch in tqdm(enumerate(testloader, start=1)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        # image: (1,3,H,W), label: (1,H,W)
        image = image.cuda()
        label = label.cuda()

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
            'volume_path': './data/COCA_3frames/test_npz',
            'list_dir': './data/COCA_3frames/lists_COCA',
            'num_classes': 4,
            'max_epochs': 300,
            'batch_size': 36,
            'base_lr': 0.00001,
            'img_size': 512,
            'encoder': 'resnet50_sa',
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
            encoder_name=args.encoder,
            encoder_weights="imagenet",
            in_channels=1,
            classes=args.num_classes,
            ).cuda()
    
    snapshot_path = f"./model/{net.__class__.__name__ + '_' + args.encoder}/{dataset_name + '_' + str(args.img_size)}/{args.exp_setting}/{'epo' + str(args.max_epochs)}"
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    
    best_model_path = glob(os.path.join(snapshot_path, '*_best_model.pth'))[0] # 유일한 best_model 파일 선택
    if not best_model_path:
        raise FileNotFoundError(f"Best model not found at {snapshot_path}")
    net.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model from: {best_model_path}")
    
    snapshot_name = snapshot_path.split('/')[-1]
    log_folder = f"./test_log/{net.__class__.__name__ + '_' + args.encoder}/{dataset_name + '_' + str(args.img_size)}/{args.exp_setting}"
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + snapshot_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(best_model_path)
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = './predictions'
        test_save_path = os.path.join(args.test_save_dir, net.__class__.__name__ + '_' + args.encoder, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    inference(args, net, test_save_path)