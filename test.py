import os
import sys
import random
import argparse
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import SimpleITK as sitk
import segmentation_models_pytorch as smp
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms as T
from datasets.dataset import COCA_dataset, Resize, ToTensor
from utils import calculate_metric_percase

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

def parse_case_and_slice_id(full_name: str):
    """
    'case0001_slice012' -> ('case0001', 12)
    'case0002'          -> ('case0002', 0)
    """
    if '_slice' in full_name:
        case_id, slice_str = full_name.split('_slice', 1)
        slice_id = int(slice_str)
    else:
        case_id = full_name
        slice_id = 0
    return case_id, slice_id

def run_inference_on_slice(
    image: torch.Tensor,
    label: torch.Tensor,
    model: torch.nn.Module,
    patch_size=(256, 256),
    test_save_path=None,
    case_name=None,
    z_spacing=1
):
    """
    슬라이스 단위로 추론하는 함수.
    (3장 슬라이스를 채널로 쌓은 2D 입력 -> 모델 예측 -> 2D 결과 반환)

    Args:
        image:  (B=1, 3, H, W) 형태의 텐서.
        label:  (B=1, H, W) 형태의 텐서.
        model:  추론할 모델(torch.nn.Module).
        patch_size: (height, width) 모델에 입력할 기본 크기.
        test_save_path: (옵션) 슬라이스별 nii.gz 저장 경로.
        case_name: (옵션) 파일 저장 시 사용될 case 식별자.
        z_spacing:  (옵션) 저장 시의 Z축 spacing 값.

    Returns:
        pred_slice (np.ndarray): (H, W), 슬라이스 예측 결과(0~N).
        label_slice (np.ndarray): (H, W), 슬라이스 라벨(정답).
    """
    image_np = image.squeeze(0).cpu().numpy()  # (3, H, W)
    label_np = label.squeeze(0).cpu().numpy()  # (H, W)

    C, H, W = image_np.shape
    assert C == 3, "Input image must have 3 channels (3-slices)."

    model.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(image_np).unsqueeze(0).float().cuda()  # (1, 3, H', W')
        logits = model(input_tensor)   # (1, num_classes, H', W') 가정
        pred_2d = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()

    prediction = pred_2d.astype(np.uint8)
    label_slice = label_np.astype(np.uint8)

    # (옵션) 슬라이스 결과 저장
    if test_save_path is not None and case_name is not None:
        img_for_save = image_np.transpose(1, 2, 0)  # (H, W, 3)
        pred_for_save = prediction
        label_for_save = label_slice

        img_itk = sitk.GetImageFromArray(img_for_save.astype(np.float32))
        pred_itk = sitk.GetImageFromArray(pred_for_save.astype(np.float32))
        label_itk = sitk.GetImageFromArray(label_for_save.astype(np.float32))

        img_itk.SetSpacing((0.375, 0.375, z_spacing))
        pred_itk.SetSpacing((0.375, 0.375, z_spacing))
        label_itk.SetSpacing((0.375, 0.375, z_spacing))

        sitk.WriteImage(pred_itk, f"{test_save_path}/{case_name}_pred.nii.gz")
        sitk.WriteImage(img_itk,  f"{test_save_path}/{case_name}_img.nii.gz")
        sitk.WriteImage(label_itk, f"{test_save_path}/{case_name}_gt.nii.gz")

    return prediction, label_slice

def accumulate_slice_prediction(pred_dict, label_dict, case_id, slice_id, pred_2d, label_2d):
    """
    dict에 슬라이스 단위 예측 결과/라벨을 누적.
    pred_dict[case_id][slice_id] = pred_2d
    label_dict[case_id][slice_id] = label_2d
    """
    if case_id not in pred_dict:
        pred_dict[case_id] = {}
        label_dict[case_id] = {}
    pred_dict[case_id][slice_id] = pred_2d
    label_dict[case_id][slice_id] = label_2d
    
def build_3d_volume(pred_slices: dict, label_slices: dict):
    """
    {slice_id: 2D pred, ...} -> 3D (H, W, depth) 배열로 변환
    """
    sorted_ids = sorted(pred_slices.keys())  # [0,1,2,...]
    min_z, max_z = sorted_ids[0], sorted_ids[-1]
    depth = max_z - min_z + 1

    # H, W 파악
    any_slice = sorted_ids[0]
    H, W = pred_slices[any_slice].shape

    pred_3d = np.zeros((H, W, depth), dtype=np.uint8)
    label_3d = np.zeros((H, W, depth), dtype=np.uint8)

    for z in sorted_ids:
        pred_3d[..., z - min_z] = pred_slices[z]
        label_3d[..., z - min_z] = label_slices[z]

    return pred_3d, label_3d


def inference(args, model, test_save_path=None):
    # 1) DataLoader 준비
    test_transform = T.Compose([
        Resize(output_size=[args.img_size, args.img_size]),
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

    # 2) (케이스 -> {슬라이스 -> (pred2D, label2D)}) 저장할 dict
    pred_slices_dict = {}
    label_slices_dict = {}

    # 3) 슬라이스 단위 추론 & dict 누적
    for i_batch, sampled_batch in tqdm(enumerate(testloader, start=1), total=len(testloader)):
        # image: (1,3,H,W), label: (1,H,W)
        image, label, full_case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        # case_id, slice_id 추출
        case_id, slice_id = parse_case_and_slice_id(full_case_name)

        # 슬라이스 추론
        pred_2d, label_2d = run_inference_on_slice(
            image, label, model,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path,
            case_name=full_case_name,
            z_spacing=args.z_spacing
        )

        # 누적
        accumulate_slice_prediction(pred_slices_dict, label_slices_dict, case_id, slice_id, pred_2d, label_2d)

    # 4) 케이스 단위로 3D 볼륨 합치고, 메트릭 계산
    metric_list_per_case = []
    case_list = sorted(pred_slices_dict.keys())

    for case_id in case_list:
        # (H, W, depth)
        pred_3d, label_3d = build_3d_volume(pred_slices_dict[case_id], label_slices_dict[case_id])

        # 클래스별 Dice, mAP, HD
        # classes 인자를 test_single_slice에서 제거했으므로,
        # 여기서는 args.num_classes를 사용 (1 ~ num_classes-1)
        metrics_this_case = []
        for c in range(1, args.num_classes):
            dice, m_ap, hd = calculate_metric_percase(pred_3d == c, label_3d == c)
            metrics_this_case.append((dice, m_ap, hd))
        
        metrics_this_case = np.array(metrics_this_case)  # (num_classes-1, 3)
        metric_list_per_case.append(metrics_this_case)

        # 케이스별 평균 로깅
        mean_dice_case = np.nanmean(metrics_this_case[:, 0])
        mean_map_case  = np.nanmean(metrics_this_case[:, 1])
        mean_hd_case   = np.nanmean(metrics_this_case[:, 2])
        logging.info(
            f"{case_id} - mean_dice: {mean_dice_case:.4f}, mean_m_ap: {mean_map_case:.4f}, mean_hd95: {mean_hd_case:.2f}"
        )

    # 5) 전체 케이스 평균
    metric_array = np.array(metric_list_per_case)  # (n_cases, num_classes-1, 3)

    # 클래스별 평균
    for c_idx in range(1, args.num_classes):
        dice_c = np.nanmean(metric_array[:, c_idx-1, 0])
        map_c  = np.nanmean(metric_array[:, c_idx-1, 1])
        hd_c   = np.nanmean(metric_array[:, c_idx-1, 2])
        logging.info(
            f"Mean class {c_idx} - dice: {dice_c:.4f}, m_ap: {map_c:.4f}, hd95: {hd_c:.2f}"
        )

    # 전체 평균
    mean_dice_all = np.nanmean(metric_array[:, :, 0])
    mean_map_all  = np.nanmean(metric_array[:, :, 1])
    mean_hd_all   = np.nanmean(metric_array[:, :, 2])
    logging.info(
        f"Testing performance - mean_dice: {mean_dice_all:.4f}, mean_m_ap: {mean_map_all:.4f}, mean_hd95: {mean_hd_all:.2f}"
    )

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
            'batch_size': 32,
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