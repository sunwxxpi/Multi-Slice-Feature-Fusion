import os
import re
import random
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import segmentation_models_pytorch as smp
from networks.emcad.networks import EMCADNet, EMCAD_SA_Net
from trainer import trainer_coca
from utils import SMP_ENCODERS, EMCAD_ENCODERS, allowed_encoders, derive_num_slices

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='COCA', help='dataset name')
parser.add_argument('--num_classes', type=int, default=5, help='output channel of network')
parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.00001, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
parser.add_argument('--encoder', type=str, default='resnet50_sa', help='encoder 이름. --decoder 에 따라 허용 목록이 다르다',
                    choices=sorted(set(SMP_ENCODERS + EMCAD_ENCODERS)))
parser.add_argument('--decoder', type=str, default='unet', choices=['unet', 'segformer', 'emcad', 'emcad_sa'])
parser.add_argument('--exp_setting', type=str,  default='default', help='이 실행의 결과가 저장될 exp_setting 이름')
parser.add_argument('--init_from', type=str, default='',
                    help='가중치를 가져올 exp_setting. 지정하면 파인튜닝이 된다 '
                         '(epo/bs/lr 이 같은 디렉터리에서 찾는다)')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=42, help='random seed')
# 5-fold CV 옵션 (기본 비활성, 단일 hold-out 경로와 하위 호환)
parser.add_argument('--use_5fold_cv', action="store_true", help='use 433-case stratified 5-fold CV')
parser.add_argument('--fold_idx', type=int, default=0, help='validation fold index (0..4)')
# 기본값은 cwd 기준 상대경로다 — 체크포인트 경로(`./model/`)와 같은 기준이라 저장소 루트에서 실행해야 한다.
parser.add_argument('--root_path_5fold', type=str, default='./data/datasets/COCA/COCA_3frames_5fold', help='5-fold per-case volume root (images/, labels/)')
parser.add_argument('--list_dir_5fold', type=str, default='./data/datasets/COCA/COCA_3frames_5fold/lists_COCA_5fold', help='5-fold list dir (fold0.txt..fold4.txt)')
parser.add_argument('--hu_stats_path', type=str, default='./data/datasets/COCA/COCA_3frames_5fold/hu_stats_433.json', help='433-case HU normalization stats json')
parser.add_argument('--early_stopping_patience', type=int, default=50, help='stop if val_loss not improved for N epochs (0=disabled)')
parser.add_argument('--early_stopping_min_delta', type=float, default=0.0, help='min val_loss improvement to reset patience')
# EMCAD 디코더 전용 (--decoder emcad/emcad_sa 에서만 사용)
parser.add_argument('--expansion_factor', type=int, default=2, help='MSCB block 의 expansion factor')
parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[1, 3, 5], help='MSDC block 의 multi-scale kernel 크기')
parser.add_argument('--lgag_ks', type=int, default=3, help='LGAG kernel 크기')
parser.add_argument('--activation_mscb', type=str, default='relu6', help='MSCB 활성함수 (relu6 | relu)')
parser.add_argument('--no_dw_parallel', action='store_true', help='depth-wise parallel convolution 비활성')
parser.add_argument('--concatenation', action='store_true', help='MSDC block 에서 feature map 을 concat')
parser.add_argument('--no_pretrain', action='store_true', help='pretrained encoder 가중치 로딩 비활성')
# 학습 레시피. 미지정 시 --decoder 계열에서 유도한다 (아래 유도 블록 참고).
parser.add_argument('--supervision', type=str, default=None, choices=['mutation', 'deep_supervision', 'last_layer'],
                    help='deep supervision 전략. 기본값은 decoder 에서 유도 (unet/segformer=last_layer, emcad*=mutation)')
parser.add_argument('--dice_weight', type=float, default=None, help='손실의 Dice 가중치. 기본값은 decoder 에서 유도 (0.5 / emcad*=0.7)')
parser.add_argument('--ce_weight', type=float, default=None, help='손실의 CE 가중치. 기본값은 decoder 에서 유도 (0.5 / emcad*=0.3)')
args = parser.parse_args()

# --decoder 별 허용 encoder 검증. argparse choices 는 합집합이라 조합 검증이 따로 필요하다.
if args.encoder not in allowed_encoders(args.decoder):
    parser.error(f"--decoder {args.decoder} 는 --encoder {args.encoder} 를 지원하지 않음. "
                 f"허용: {allowed_encoders(args.decoder)}")

# fold 정체성은 경로에 안 들어가고 손으로 친 exp_setting 문자열이 전부다.
# 불일치를 경고로 두면 fold1 학습이 fold0 체크포인트 디렉터리를 덮어쓴다.
if f"fold{args.fold_idx}" not in args.exp_setting:
    parser.error(f"--fold_idx={args.fold_idx} 인데 --exp_setting='{args.exp_setting}' 에 "
                 f"'fold{args.fold_idx}' 가 없음. 체크포인트 경로에 fold 가 반영되지 않아 덮어쓸 위험.")

# 다른 fold 의 체크포인트에서 출발하면 그 가중치는 지금의 val fold 를 이미 학습에 썼다 —
# 검증이 오염된다. fold 토큰이 없는 이름(다른 코호트)은 대조할 fold 가 없어 통과시킨다.
if args.init_from and re.search(r'fold\d+', args.init_from) and f"fold{args.fold_idx}" not in args.init_from:
    parser.error(f"--fold_idx={args.fold_idx} 인데 --init_from='{args.init_from}' 는 다른 fold 의 "
                 f"체크포인트다. 출발 가중치가 fold{args.fold_idx} 케이스를 이미 학습해 검증이 오염된다.")

args.num_slices = derive_num_slices(args.decoder, args.encoder)

# 학습 레시피 기본값을 decoder 계열에서 유도. 명시 시 그 값을 존중한다.
_is_emcad = args.decoder in ('emcad', 'emcad_sa')
if args.supervision is None:
    args.supervision = 'mutation' if _is_emcad else 'last_layer'
if args.dice_weight is None:
    args.dice_weight = 0.7 if _is_emcad else 0.5
if args.ce_weight is None:
    args.ce_weight = 0.3 if _is_emcad else 0.5

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
                       encoder_weights=None if args.no_pretrain else "imagenet",
                       in_channels=1,
                       classes=args.num_classes).cuda()
    elif args.decoder == 'segformer':
        net = smp.Segformer(encoder_name=args.encoder,
                            encoder_weights=None if args.no_pretrain else "imagenet",
                            in_channels=1,
                            classes=args.num_classes).cuda()
    else:
        NetCls = EMCAD_SA_Net if args.decoder == 'emcad_sa' else EMCADNet
        net = NetCls(num_classes=args.num_classes,
                     kernel_sizes=args.kernel_sizes,
                     expansion_factor=args.expansion_factor,
                     dw_parallel=not args.no_dw_parallel,
                     add=not args.concatenation,
                     lgag_ks=args.lgag_ks,
                     activation=args.activation_mscb,
                     encoder=args.encoder,
                     pretrain=not args.no_pretrain).cuda()

    # from torchinfo import summary
    # torchinfo_summary = str(summary(net, input_size=(args.batch_size, args.num_slices, args.img_size, args.img_size),
    #                                 col_width=20, depth=5,
    #                                 row_settings=["depth", "var_names"],
    #                                 col_names=["input_size", "kernel_size", "output_size", "params_percent"]))
    # output_file = f"{net.__class__.__name__}_{args.encoder}_model_summary.txt"
    # with open(output_file, "w") as file:
    #     file.write(torchinfo_summary)

    exp_path = os.path.join(net.__class__.__name__ + '_' + args.encoder, args.dataset + '_' + str(args.img_size), args.exp_setting)
    parameter_path = 'epo' + str(args.max_epochs) + '_bs' + str(args.batch_size) + '_lr' + str(args.base_lr)
    snapshot_path = os.path.join("./model/", exp_path, parameter_path)
    os.makedirs(snapshot_path, exist_ok=True)
    
    # 파인튜닝: --init_from 의 best 체크포인트에서 출발한다. 저장은 위 snapshot_path 그대로라
    # 원본 체크포인트는 건드리지 않는다.
    if args.init_from:
        init_exp_path = os.path.join(net.__class__.__name__ + '_' + args.encoder, args.dataset + '_' + str(args.img_size), args.init_from)
        init_path = os.path.join("./model/", init_exp_path, parameter_path)
        if not os.path.isdir(init_path):
            raise FileNotFoundError("--init_from 경로가 없음: " + init_path)

        best_model_file = None
        for f in os.listdir(init_path):
            name, ext = os.path.splitext(f)
            if name.endswith("best_model"):
                best_model_file = f
                break
        if best_model_file is None:
            raise FileNotFoundError("No checkpoint ending with 'best_model' found in " + init_path)

        checkpoint_path = os.path.join(init_path, best_model_file)
        checkpoint = torch.load(checkpoint_path)
        # Remove segmentation head weights to avoid size mismatch (checkpoint was trained for 5 classes)
        # head 키 이름은 decoder 계열마다 다르다 (SMP=segmentation_head. / EMCAD=out_head1~4).
        head_prefix = "segmentation_head." if args.decoder in ('unet', 'segformer') else "out_head"
        for key in list(checkpoint.keys()):
            if key.startswith(head_prefix):
                del checkpoint[key]
        net.load_state_dict(checkpoint, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")

    trainer = {'COCA': trainer_coca}
    trainer[args.dataset](args, net, snapshot_path)