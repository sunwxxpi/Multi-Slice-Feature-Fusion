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
from tester import inference, get_attn_hook

def add_encoder_prefix(state_dict, prefix='encoder.'):
    new_state_dict = {}
    for key, value in state_dict.items():
        # key가 'encoder.', 'decoder.', 'segmentation_head.'로 시작하지 않으면 접두어 추가
        if not key.startswith(("encoder.", "decoder.", "segmentation_head.")):
            new_key = prefix + key
        else:
            new_key = key
        new_state_dict[new_key] = value
        
    return new_state_dict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='COCA', help='dataset name')
parser.add_argument('--root_path', type=str, default='/home/psw/AVS-Diagnosis/COCA/COCA_3frames/test_npz', help='root dir for validation volume data')
parser.add_argument('--list_dir', type=str, default='/home/psw/AVS-Diagnosis/COCA/COCA_3frames/lists_COCA', help='list dir')
parser.add_argument('--num_classes', type=int, default=5, help='output channel of network')
parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.00001, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
parser.add_argument('--encoder', type=str, default='resnet50_sa', help='for segmentation_models_pytorch encoder', choices=['resnet50_sa', 'densenet201_sa', 'efficientnet-b4_sa', 'mit_b2_sa'])
parser.add_argument('--decoder', type=str, default='unet', help='for segmentation_models_pytorch decoder', choices=['unet', 'segformer'])
parser.add_argument('--exp_setting', type=str,  default='default', help='description of experiment setting')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--z_spacing', type=int, default=3, help='z spacing of the volume')
# 5-fold CV 옵션 (기본 비활성, 단일 hold-out 경로와 하위 호환). 평가 대상은 fold_idx (= val fold).
parser.add_argument('--use_5fold_cv', action="store_true", help='evaluate on 5-fold validation fold')
parser.add_argument('--fold_idx', type=int, default=0, help='validation fold index to evaluate (0..4)')
parser.add_argument('--root_path_5fold', type=str, default='/home/psw/AVS-Diagnosis/COCA/COCA_3frames_5fold', help='5-fold per-case volume root (images/, labels/)')
parser.add_argument('--list_dir_5fold', type=str, default='/home/psw/AVS-Diagnosis/COCA/COCA_3frames_5fold/lists_COCA_5fold', help='5-fold list dir (fold0.txt..fold4.txt)')
parser.add_argument('--hu_stats_path', type=str, default='/home/psw/AVS-Diagnosis/COCA/COCA_3frames_5fold/hu_stats_433.json', help='433-case HU normalization stats json')
# Attention 시각화 옵션. 켜면 NonLocalBlock 들의 return_attention=True 자동 토글 + hook 자동 등록 + 시각화 저장.
parser.add_argument('--save_attention', action="store_true", help='enable attention visualization saving')
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
                       encoder_weights=None,
                       in_channels=1,
                       classes=args.num_classes).cuda()
    elif args.decoder == 'segformer':
        net = smp.Segformer(encoder_name=args.encoder,
                            encoder_weights=None,
                            in_channels=1,
                            classes=args.num_classes).cuda()
    
    exp_path = os.path.join(net.__class__.__name__ + '_' + args.encoder, args.dataset + '_' + str(args.img_size), args.exp_setting)
    parameter_path = 'epo' + str(args.max_epochs) + '_bs' + str(args.batch_size) + '_lr' + str(args.base_lr)
    
    snapshot_path = os.path.join("./model/", exp_path, parameter_path)
    best_model_path = glob(os.path.join(snapshot_path, '*_best_model.pth'))[0]
    if not best_model_path:
        raise FileNotFoundError(f"Best model not found at {snapshot_path}")
    
    checkpoint = torch.load(best_model_path, map_location='cpu')
    fixed_state_dict = add_encoder_prefix(checkpoint, prefix="encoder.")
    
    net.load_state_dict(fixed_state_dict)
    print(f"\nLoaded best model from: {best_model_path}")
    
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
    # 백본별 attribute 명 다른 문제 (densenet/efficientnet/mit) 회피 — `return_attention` 속성 보유 모듈을 모두 잡는다.
    # hook 키는 visualize_attention 이 기대하는 stage{N}_{prev|self|next} 형식으로 변환한다
    # (모듈명 cross_attention_prev_3 → stage3_prev). 패턴 불일치 시 모듈명을 그대로 쓴다.
    if args.save_attention and hasattr(net, 'encoder'):
        import re
        hook_count = 0
        for module_name, module in net.encoder.named_modules():
            if hasattr(module, 'return_attention'):
                module.return_attention = True
                m = re.search(r'(prev|self|next).*?(\d+)$', module_name)
                attn_key = f"stage{m.group(2)}_{m.group(1)}" if m else module_name
                module.register_forward_hook(get_attn_hook(attn_key))
                hook_count += 1
        print(f"Registered attention hooks on {hook_count} NonLocalBlock(s).")

    # 추론 실행
    inference(args, net, test_save_path)