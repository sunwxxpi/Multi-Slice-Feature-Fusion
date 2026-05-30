import os
import sys
import random
import argparse
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from glob import glob
from networks.emcad.networks import EMCAD_SA_Net
from tester import inference, get_attn_hook

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='COCA', help='dataset name')
parser.add_argument('--root_path', type=str, default='/home/psw/AVS-Diagnosis/COCA/COCA_3frames/test_npz', help='root dir for validation volume data')
parser.add_argument('--list_dir', type=str, default='/home/psw/AVS-Diagnosis/COCA/COCA_3frames/lists_COCA', help='list dir')
parser.add_argument('--num_classes', type=int, default=5, help='output channel of network')
parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.00001, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
parser.add_argument('--exp_setting', type=str,  default='default', help='description of experiment setting')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--save_attention', action="store_true", help='enable attention visualization saving')
parser.add_argument('--z_spacing', type=int, default=3, help='z spacing of the volume')
# 5-fold CV 옵션 (기본 비활성, 단일 hold-out 경로와 하위 호환)
parser.add_argument('--use_5fold_cv', action="store_true", help='use 433-case stratified 5-fold CV')
parser.add_argument('--fold_idx', type=int, default=0, help='validation fold index (0..4) = 평가 셋')
parser.add_argument('--root_path_5fold', type=str, default='/home/psw/AVS-Diagnosis/COCA/COCA_3frames_5fold', help='5-fold per-case volume root (images/, labels/)')
parser.add_argument('--list_dir_5fold', type=str, default='/home/psw/AVS-Diagnosis/COCA/COCA_3frames_5fold/lists_COCA_5fold', help='5-fold list dir (fold0.txt..fold4.txt)')
parser.add_argument('--hu_stats_path', type=str, default='/home/psw/AVS-Diagnosis/COCA/COCA_3frames_5fold/hu_stats_433.json', help='433-case HU normalization stats json')

# network related parameters
parser.add_argument('--encoder', type=str,
                    default='pvt_v2_b2', help='Name of encoder: pvt_v2_b2, pvt_v2_b0, resnet18, resnet34 ...')
parser.add_argument('--expansion_factor', type=int,
                    default=2, help='expansion factor in MSCB block')
parser.add_argument('--kernel_sizes', type=int, nargs='+',
                    default=[1, 3, 5], help='multi-scale kernel sizes in MSDC block')
parser.add_argument('--lgag_ks', type=int,
                    default=3, help='Kernel size in LGAG')
parser.add_argument('--activation_mscb', type=str,
                    default='relu6', help='activation used in MSCB: relu6 or relu')
parser.add_argument('--no_dw_parallel', action='store_true',
                    default=False, help='use this flag to disable depth-wise parallel convolutions')
parser.add_argument('--concatenation', action='store_true',
                    default=False, help='use this flag to concatenate feature maps in MSDC block')
parser.add_argument('--no_pretrain', action='store_true',
                    default=False, help='use this flag to turn off loading pretrained enocder weights')
parser.add_argument('--supervision', type=str,
                    default='mutation', help='loss supervision: mutation, deep_supervision or last_layer')

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

    net = EMCAD_SA_Net(num_classes=args.num_classes, 
                   kernel_sizes=args.kernel_sizes, 
                   expansion_factor=args.expansion_factor, 
                   dw_parallel=not args.no_dw_parallel, 
                   add=not args.concatenation, 
                   lgag_ks=args.lgag_ks, 
                   activation=args.activation_mscb, 
                   encoder=args.encoder, 
                   pretrain=not args.no_pretrain).cuda()
    
    exp_path = os.path.join(net.__class__.__name__ + '_' + args.encoder, args.dataset + '_' + str(args.img_size), args.exp_setting)
    parameter_path = 'epo' + str(args.max_epochs) + '_bs' + str(args.batch_size) + '_lr' + str(args.base_lr)
    
    snapshot_path = os.path.join("./model/", exp_path, parameter_path)
    best_model_path = glob(os.path.join(snapshot_path, '*_best_model.pth'))[0]
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

    # --save_attention 시 NonLocalBlock 자동 검색 → return_attention=True 토글 + hook 등록.
    # EMCAD-SA 의 NonLocalBlock 은 net.backbone 아래(cross_attention_*_{3,4}) 에 있으므로
    # net 전체를 순회해 `return_attention` 속성 보유 모듈을 모두 잡는다.
    # hook 키는 visualize_attention 이 기대하는 stage{N}_{prev|self|next} 형식으로 변환한다
    # (모듈명 backbone.cross_attention_prev_3 → stage3_prev). 패턴 불일치 시 모듈명을 그대로 쓴다.
    if args.save_attention:
        import re
        hook_count = 0
        for module_name, module in net.named_modules():
            if hasattr(module, 'return_attention'):
                module.return_attention = True
                m = re.search(r'(prev|self|next).*?(\d+)$', module_name)
                attn_key = f"stage{m.group(2)}_{m.group(1)}" if m else module_name
                module.register_forward_hook(get_attn_hook(attn_key))
                hook_count += 1
        print(f"Registered attention hooks on {hook_count} NonLocalBlock(s).")

    inference(args, net, test_save_path)