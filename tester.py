import os
import logging
import math
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms as T
from matplotlib.patches import Rectangle
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric, SurfaceDistanceMetric
from datasets.dataset import COCA_dataset, COCAVolumeDataset, load_hu_stats, Resize, ToTensor

# 전역 딕셔너리: forward hook을 통해 각 NonLocalBlock의 attention map을 저장
attn_dict = {}

def get_attn_hook(name):
    """
    각 NonLocalBlock의 forward 시, (z, attention_weights) 튜플에서 attention_weights를 attn_dict에 저장하는 hook 함수.
    """
    def hook(module, input, output):
        # output: (z, attention_weights) → attention_weights: (B, num_heads, N, N)
        attn_dict[name] = output[1].detach().cpu()
        
    return hook

def parse_case_and_slice_id(full_name: str) -> tuple:
    if '_slice' in full_name:
        case_id, slice_str = full_name.split('_slice', 1)
        slice_id = int(slice_str)
    else:
        case_id = full_name
        slice_id = 0
        
    return case_id, slice_id

def run_inference_on_slice(image: torch.Tensor, label: torch.Tensor, model: torch.nn.Module) -> tuple:
    """
    image: (1, 3, H, W) 텐서, 3채널은 prev, main, next 슬라이스
    label: (1, H, W) 텐서 (center slice의 레이블)
    """
    image_np = image.squeeze(0).cpu().numpy()  # (3, H, W)
    label_np = label.squeeze(0).cpu().numpy()    # (H, W)
    C, H, W = image_np.shape
    assert C == 3, "Input image must have 3 channels (3 slices)."

    model.eval()
    with torch.no_grad():
        with autocast():
            input_tensor = image.float().cuda()
            logits = model(input_tensor)
            pred_2d = torch.argmax(torch.softmax(logits[3], dim=1), dim=1).squeeze(0).cpu().numpy()

    prediction = pred_2d.astype(np.uint8)
    label_slice = label_np.astype(np.uint8)
    
    return prediction, label_slice

def accumulate_slice_prediction(image_dict: dict, pred_dict: dict, label_dict: dict, case_id: str, slice_id: int, pred_2d: np.ndarray, label_2d: np.ndarray, img_2d: np.ndarray):
    if case_id not in pred_dict:
        image_dict[case_id] = {}
        pred_dict[case_id] = {}
        label_dict[case_id] = {}
        
    image_dict[case_id][slice_id] = img_2d
    pred_dict[case_id][slice_id] = pred_2d
    label_dict[case_id][slice_id] = label_2d

def build_3d_volume(image_slices: dict, pred_slices: dict, label_slices: dict, case_id, args, test_save_path) -> tuple:
    sorted_ids = sorted(pred_slices.keys())
    min_z, max_z = sorted_ids[0], sorted_ids[-1]
    depth = max_z - min_z + 1

    any_slice = sorted_ids[0]
    H, W = pred_slices[any_slice].shape

    image_3d = np.zeros((depth, H, W), dtype=np.uint8)
    pred_3d = np.zeros((depth, H, W), dtype=np.uint8)
    label_3d = np.zeros((depth, H, W), dtype=np.uint8)

    for z in sorted_ids:
        index = z - min_z
        image_3d[index, :, :] = image_slices[z][1, :, :]
        pred_3d[index, :, :] = pred_slices[z]
        label_3d[index, :, :] = label_slices[z]
        
    if args.is_savenii and test_save_path:
        for array, suffix in zip([image_3d, pred_3d, label_3d], ["img", "pred", "gt"]):
            array = np.flip(np.transpose(array, (0, 2, 1)), (1, 2))
            arr_itk = sitk.GetImageFromArray(array.astype(np.float32))
            arr_itk.SetSpacing((0.375, 0.375, args.z_spacing))
            sitk.WriteImage(arr_itk, f"{test_save_path}/{case_id}_{suffix}.nii.gz")

    return image_3d, pred_3d, label_3d

def compute_metrics_3d(pred_3d, label_3d, num_classes, dice_metric, miou_metric, hd_metric, case_id):
    pred_tensor = torch.from_numpy(pred_3d).unsqueeze(0).unsqueeze(0).float().cuda()
    label_tensor = torch.from_numpy(label_3d).unsqueeze(0).unsqueeze(0).float().cuda()
    
    D, H, W = pred_3d.shape  # 3D 볼륨 크기
    max_hd_3d = math.sqrt((D-1)**2 + (H-1)**2 + (W-1)**2)  # 3D 공간의 대각선 거리
    
    metrics_per_class = []
    
    logging.info(f"Metrics for Case: {case_id}")
    for c in range(1, num_classes):
        pred_class = (pred_tensor == c).float()
        label_class = (label_tensor == c).float()
        
        dice = dice_metric(pred_class, label_class)
        miou = miou_metric(pred_class, label_class)
        hd = hd_metric(pred_class, label_class)
        
        if isinstance(hd, torch.Tensor):
            hd = hd.item()

        gt_sum = label_class.sum()
        pred_sum = pred_class.sum()

        if gt_sum == 0 and pred_sum == 0:
            status = "(GT==0 & Pred==0)"
        elif gt_sum == 0 and pred_sum > 0:
            hd = np.nan
            status = "(GT==0 & Pred>0)"
        elif gt_sum > 0 and pred_sum == 0:
            hd = max_hd_3d
            status = "(GT>0 & Pred==0)"
        else:
            status = "(GT>0 & Pred>0)"

        logging.info(f"  Class {c}: Dice: {dice.item():.4f}, mIoU: {miou.item():.4f}, HD: {hd:.2f} {status}")
        
        metrics_per_class.append((dice.item(), miou.item(), hd))
    
    metrics_per_class = np.array(metrics_per_class)
    mean_dice = np.nanmean(metrics_per_class[:, 0])
    mean_iou = np.nanmean(metrics_per_class[:, 1])
    mean_hd = np.nanmean(metrics_per_class[:, 2])
    
    logging.info(f"  [Case {case_id}] - Mean Dice: {mean_dice:.4f}, Mean mIoU: {mean_iou:.4f}, Mean HD: {mean_hd:.2f}\n")
    
    return metrics_per_class

def log_3d_metrics(metric_array, num_classes):
    logging.info("\nOverall 3D Metrics Across All Cases:")

    class_dice_means = []
    class_miou_means = []
    class_hd_means = []

    for c_idx in range(1, num_classes):
        dice_c = np.nanmean(metric_array[:, c_idx-1, 0])
        miou_c = np.nanmean(metric_array[:, c_idx-1, 1])
        hd_c = np.nanmean(metric_array[:, c_idx-1, 2])

        logging.info(f"  [3D] Class {c_idx} - Dice: {dice_c:.4f}, mIoU: {miou_c:.4f}, HD: {hd_c:.2f}")

        class_dice_means.append(dice_c)
        class_miou_means.append(miou_c)
        class_hd_means.append(hd_c)

    mean_dice_all = np.nanmean(class_dice_means)
    mean_miou_all = np.nanmean(class_miou_means)
    mean_hd_all = np.nanmean(class_hd_means)

    logging.info(f"  [3D] Testing Performance - Mean Dice: {mean_dice_all:.4f}, Mean mIoU: {mean_miou_all:.4f}, Mean HD: {mean_hd_all:.2f}")

# 시각화 표시 방향. COCA 원본 axial 은 옆으로 누운 cardiac FOV 라, 사람이 보는
# 표준 방향(척추 아래)으로 맞추려면 왼쪽 90°(CCW) 1회 회전이 필요하다.
# 0 으로 두면 회전 없이 원본 array 방향으로 표시(기존 동작).
VIS_ROT90_CCW = 1

def _rot90_cell(r, c, g, k):
    # 정사각 그리드(side g)에서 CCW 90° k회 회전 시 (row,col) 좌표 변환.
    # CCW 1회: (r,c) -> (g-1-c, r). 이미지·박스·query 모두 같은 식으로 옮긴다.
    k %= 4
    for _ in range(k):
        r, c = g - 1 - c, r
    return r, c

def visualize_attention(attn_dict, input_image, label, file_name, save_path):
    """
    attn_dict: forward hook을 통해 저장된 attention map 딕셔너리  
               (키: "stage3_prev", "stage3_self", "stage3_next", 
                     "stage4_prev", "stage4_self", "stage4_next")
               각 값은 (B, num_heads, N, N) 텐서.
    input_image: 원본 입력 이미지 (numpy, shape: (3, H, W); 채널 0: prev, 1: center, 2: next)
    label: center slice의 segmentation label (numpy, shape: (H, W))
    file_name: 시각화 결과 저장 시 사용할 파일명 식별자 (예: 케이스 및 slice 이름)
    save_path: 시각화 이미지가 저장될 디렉토리 경로

    수행 과정:
      1. center slice(main_img)와 label을 그대로 사용.
      2. label에서 병변 영역(픽셀 값 > 0)을 확인하여, GT가 없는 경우엔 계산/저장을 건너뛰고,
         병변 영역이 있다면 그 평균 좌표를 center slice의 query 위치(병변 영역)로 선정.
      3. 각 attention map에 대해, 해당 query 위치(q_index)를 기준으로 
         query에 해당하는 row(attn[:, :, q_index, :])의 attention 분포를 추출하여 head 평균 후,
         원래 feature map 해상도로 복원하고, 최종적으로 입력 이미지 해상도(W, H)로 업샘플링.
      4. 업샘플된 attention map에 grid overlay를 적용하고, q_index 기반의 grid cell 위치를 빨간색 사각형으로 강조.
      5. stage3와 stage4 각각에 대해 prev, self, next 총 3가지 map을 2행 3열 subplot으로 그려 저장.
    """
    # 1. center slice와 label을 그대로 사용
    main_img = input_image[1]  # center slice, shape: (H, W)
    H, W = main_img.shape

    # 2. label에서 병변 영역(픽셀 값 > 0) 확인
    coords = np.argwhere(label > 0)
    lesion_exists = len(coords) > 0

    # GT(ground truth)가 없는 경우엔 계산 및 이미지 저장하지 않음.
    if not lesion_exists:
        return

    # 3. 병변 영역이 존재하면, 해당 영역의 모든 좌표의 평균을 center slice의 query 위치로 선정
    q_row = int(np.mean(coords[:, 0]))
    q_col = int(np.mean(coords[:, 1]))
    q_coord = (q_row, q_col)  # 내부 계산용 (원본 array 좌표 — attention 추출은 이 방향 유지)
    # 표시용 회전(VIS_ROT90_CCW)을 query 픽셀좌표에도 적용해 제목·박스를 화면과 일치시킨다.
    # imshow(origin='upper') array 좌표(row 아래로 증가)를 그대로 쓴다 (이전: H-row y-flip 으로 박스와 상하 엇갈림).
    qr_disp, qc_disp = _rot90_cell(q_row, q_col, H, VIS_ROT90_CCW)  # COCA 512 정사각(H==W) 가정
    query_text = f"(Query col,row: ({qc_disp}, {qr_disp}))"

    # 4. 각 stage별 시각화 대상 key 설정
    stage3_keys = ["stage3_prev", "stage3_self", "stage3_next"]
    stage4_keys = ["stage4_prev", "stage4_self", "stage4_next"]

    # 2행 3열 subplot 생성 (첫 행: stage3, 두 번째 행: stage4)
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # 5a. 1-pass: 각 패널의 attention 분포를 uniform(1/N) 대비 배수로 변환한다.
    #     기존 per-panel min-max 정규화는 거의 균일한 분포까지 풀스케일 jet 로 그려 핫스팟처럼
    #     보이게 했다. uniform 대비 배수(1.0 = 무선호)로 바꾸면 평평한 패널은 평평하게,
    #     실제 집중(예: 인접 슬라이스 동일 위치)만 튀어 query↔hotspot 비교가 정직해진다.
    #     각 패널의 peak 배수와 query 위치(빨간 박스) 배수를 함께 기록해 제목에 표기한다
    #     ("query 가 박스 위치를 보는가" = attn@box, "어디든 가장 강한 집중" = peak).
    panel = {}  # key -> (ratio_resized_rot, (rect_x,rect_y), cell_w, cell_h, grid_count, r_box, r_peak)
    panel_maxes = []
    for stage_keys, grid_count in zip([stage3_keys, stage4_keys], [32, 16]):
        for key in stage_keys:
            if key not in attn_dict:
                continue
            attn = attn_dict[key]  # (B, num_heads, N, N)
            N = attn.shape[-1]
            feat = int(math.sqrt(N))  # H_feat == W_feat

            # 병변 query 위치를 feature 해상도로 환산해 q_index 산출
            q_row_feat = int(q_coord[0] / (H / feat))
            q_col_feat = int(q_coord[1] / (W / feat))
            q_index = q_row_feat * feat + q_col_feat

            # query row 의 key 분포(합=1)를 uniform 대비 배수로 변환 후 표시 방향으로 회전
            dist = attn[:, :, q_index, :].mean(dim=1)[0].numpy()  # (N,)
            ratio = (dist * N).reshape(feat, feat)                # 1.0 = uniform
            r_box = float(ratio[q_row_feat, q_col_feat])          # query(박스) 위치 배수
            r_peak = float(ratio.max())                           # 가장 강한 집중 배수
            ratio_resized = np.rot90(cv2.resize(ratio, (W, H)), VIS_ROT90_CCW)
            panel_maxes.append(r_peak)

            # 박스(query grid cell)도 표시 회전과 동일하게 옮긴다
            qrg, qcg = _rot90_cell(q_index // grid_count, q_index % grid_count, grid_count, VIS_ROT90_CCW)
            cell_w = W / grid_count
            cell_h = H / grid_count
            panel[key] = (ratio_resized, (qcg * cell_w, qrg * cell_h), cell_w, cell_h, grid_count, r_box, r_peak)

    # 공통 vmax: self-attention 의 off-query 극단 peak(수백 배) 하나가 스케일을 독식해 cross
    # 패널의 의미 있는 peak(수십 배)를 눌러버리는 것을 막기 위해 per-panel peak 의 median 을
    # 쓰고, 모든 패널이 평평한 경우(전부 ~1x)를 위해 하한 5.0 을 둔다.
    vmax = max(5.0, float(np.median(panel_maxes))) if panel_maxes else 5.0

    # 5b. 2-pass: 공통 vmin=0·vmax 로 그려 6패널 강도를 직접 비교 가능하게 한다.
    main_disp = np.rot90(main_img, VIS_ROT90_CCW)
    im = None
    for row_idx, stage_keys in enumerate([stage3_keys, stage4_keys]):
        for col_idx, key in enumerate(stage_keys):
            ax = axs[row_idx, col_idx]
            ax.axis('off')
            if key not in panel:
                ax.set_title(f"{key} not available", fontsize=10)
                continue
            ratio_resized, (rect_x, rect_y), cell_w, cell_h, grid_count, r_box, r_peak = panel[key]

            ax.imshow(main_disp, cmap='gray')
            im = ax.imshow(ratio_resized, cmap='jet', alpha=0.5, vmin=0.0, vmax=vmax)
            # attn@box = query 가 빨간 박스(=병변 자기 위치)를 보는 배수, peak = 어디든 최대 집중 배수.
            ax.set_title(f"{key}   attn@box={r_box:.0f}x  peak={r_peak:.0f}x", fontsize=10)

            for i in range(1, grid_count):
                ax.axvline(i * cell_w, color='white', linewidth=1, alpha=0.8)
                ax.axhline(i * cell_h, color='white', linewidth=1, alpha=0.8)
            ax.add_patch(Rectangle((rect_x, rect_y), cell_w, cell_h,
                                   edgecolor='red', facecolor='none', linewidth=2))

    if im is not None:
        # colorbar 축 = uniform 대비 배수 (1.0 = 무선호, >1 = query 가 해당 key 를 집중 참조).
        # vmax 초과(self 극단 peak 등)는 최상위 색으로 saturate.
        cbar = fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.025, pad=0.02, extend='max')
        cbar.set_label("attention / uniform (1.0 = no preference)", fontsize=9)
    fig.suptitle(f"{file_name}   {query_text}", fontsize=11)
    save_file = os.path.join(save_path, f"{file_name}_attn_query.png")
    plt.savefig(save_file, dpi=300)
    plt.close(fig)

def inference(args, model, test_save_path: str = None):
    test_transform = T.Compose([Resize(output_size=[args.img_size, args.img_size]),
                                ToTensor()])
    if getattr(args, 'use_5fold_cv', False):
        # 평가 대상은 학습 때의 validation fold (fold_idx). per-case 볼륨에서 로드.
        hu = load_hu_stats(args.hu_stats_path)
        with open(os.path.join(args.list_dir_5fold, f"fold{args.fold_idx}.txt"), 'r') as f:
            test_samples = [ln.strip() for ln in f if ln.strip()]
        db_test = COCAVolumeDataset(os.path.join(args.root_path_5fold, 'images'),
                                    os.path.join(args.root_path_5fold, 'labels'),
                                    test_samples, transform=test_transform, hu_stats=hu)
    else:
        db_test = COCA_dataset(base_dir=args.root_path,
                               list_dir=args.list_dir,
                               split="test",
                               transform=test_transform)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f"{len(testloader)} test iterations per epoch")

    image_slices_dict = {}
    pred_slices_dict = {}
    label_slices_dict = {}

    # attention 시각화 이미지를 저장할 폴더 설정 (--save_attention 켜진 경우에만 mkdir).
    # fallback 경로에 exp_setting 포함 — 여러 run 의 시각화가 한 폴더에 섞이는 문제 방지.
    save_attention = getattr(args, 'save_attention', False)
    attn_vis_dir = None
    if save_attention:
        if test_save_path:
            attn_vis_dir = os.path.join(test_save_path, "attention_vis")
        else:
            attn_vis_dir = os.path.join("./test_log", "attention_vis_fallback", args.exp_setting)
        os.makedirs(attn_vis_dir, exist_ok=True)

    for sampled_batch in tqdm(testloader, total=len(testloader), desc="Inference"):
        # 매 slice 마다 누수 방지 — hook 만 켜고 시각화 OFF 인 조합에서도 RAM 폭발 차단.
        attn_dict.clear()

        image, label, full_case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        case_id, slice_id = parse_case_and_slice_id(full_case_name)
        img_2d = image.squeeze(0).cpu().numpy()
        pred_2d, label_2d = run_inference_on_slice(image, label, model)

        if attn_vis_dir is not None:
            # query 픽셀(GT lesion 평균 좌표) 기반 attention 시각화. GT 없으면 visualize_attention 내부에서 skip.
            visualize_attention(attn_dict, img_2d, label_2d, full_case_name, attn_vis_dir)

        accumulate_slice_prediction(image_slices_dict, pred_slices_dict, label_slices_dict, case_id, slice_id, pred_2d, label_2d, img_2d)

    dice_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=True)
    miou_metric = MeanIoU(include_background=True, reduction="mean", ignore_empty=True)
    # hd_metric = HausdorffDistanceMetric(include_background=True, distance_metric="euclidean", percentile=95)
    hd_metric = SurfaceDistanceMetric(include_background=True, symmetric=True, distance_metric="euclidean")

    metric_list_per_case = []
    
    case_list = sorted(pred_slices_dict.keys())
    for case_id in case_list:
        _, pred_3d, label_3d = build_3d_volume(image_slices_dict[case_id], pred_slices_dict[case_id], label_slices_dict[case_id], case_id, args, test_save_path)
        metrics_per_case = compute_metrics_3d(pred_3d, label_3d, args.num_classes, dice_metric, miou_metric, hd_metric, case_id)
        metric_list_per_case.append(metrics_per_case)

    metric_array = np.array(metric_list_per_case)
    log_3d_metrics(metric_array, args.num_classes)

    return "Testing Finished!"