import os
import json
import sys
import random
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm
from utils import build_supervision, PolyLRScheduler, DiceLoss
from training_protocol import TrainingLossPlateau
from dataset import (shuffle_within_batch, COCAVolumeDataset,
                              load_hu_stats, RandomAugmentation, Resize, ToTensor)

def _read_fold_list(list_dir, k):
    with open(os.path.join(list_dir, f"fold{k}.txt"), 'r') as f:
        return [ln.strip() for ln in f if ln.strip()]

def trainer_coca(args, model, snapshot_path):
    artifacts = [name for name in os.listdir(snapshot_path)
                 if name.endswith('.pth') or name in ('log.txt', 'training_record.json')]
    if artifacts:
        raise FileExistsError('기존 학습 기록이 있습니다. 새 --exp_setting을 사용하세요: ' + snapshot_path)
    logging.basicConfig(filename=snapshot_path + "/log.txt", 
                        level=logging.INFO, 
                        format='[%(asctime)s.%(msecs)03d] %(message)s', 
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    base_lr = args.base_lr
    batch_size = args.batch_size
    
    train_transform = T.Compose([RandomAugmentation(),
                                 Resize(output_size=[args.img_size, args.img_size]),
                                 ToTensor()])
    # 테스트 fold의 목록과 이미지는 학습 중 읽지 않는다.
    hu = load_hu_stats(args.hu_stats_path)
    image_dir = os.path.join(args.root_path_5fold, 'images')
    label_dir = os.path.join(args.root_path_5fold, 'labels')
    train_samples = []
    for k in range(5):
        if k == args.fold_idx:
            continue
        train_samples += _read_fold_list(args.list_dir_5fold, k)
    db_train = COCAVolumeDataset(image_dir, label_dir, train_samples,
                                 transform=train_transform, hu_stats=hu,
                                 num_slices=args.num_slices)
    if len(db_train) == 0:
        raise ValueError('학습 데이터가 비어 있습니다.')
    logging.info(f"5-fold CV: test fold={args.fold_idx}, train folds={[k for k in range(5) if k != args.fold_idx]}, num_slices={args.num_slices}")
    logging.info('Training mode: %s; configured epochs: %d',
                 'training_loss_pilot' if args.training_loss_pilot else 'fixed_epochs', args.max_epochs)
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=False, 
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn, 
                             collate_fn=shuffle_within_batch)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    dice_loss_class = DiceLoss()
    ce_loss_class = CrossEntropyLoss()

    # deep supervision 조합. 모델 출력 개수를 알아야 하므로 첫 batch 에서 확정한다.
    ss = None
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    
    max_iterations = args.max_epochs * len(trainloader)
    scheduler = PolyLRScheduler(optimizer, initial_lr=base_lr, max_steps=max_iterations)
    
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    iter_num = 0
    max_epoch = args.max_epochs
    plateau = TrainingLossPlateau(args.training_loss_patience, args.training_loss_min_delta) if args.training_loss_pilot else None

    scaler = GradScaler()
    
    for epoch_num in tqdm(range(1, max_epoch + 1), ncols=70):
        train_dice_loss = 0.0
        train_ce_loss = 0.0
        train_loss = 0.0
        
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader, start=1):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            with autocast():
                P = model(image_batch)
                if not isinstance(P, (list, tuple)):
                    P = [P]

                if ss is None:
                    ss = build_supervision(args.supervision, len(P))
                    logging.info(f"Supervision strategy: {args.supervision} (n_outs={len(P)}) -> {ss}")

                sum_dice_loss = 0.0
                sum_ce_loss = 0.0
                loss = 0.0
                for s in ss:
                    iout = sum(P[idx] for idx in s)
                    dice_loss = dice_loss_class(iout, label_batch, softmax=True)
                    ce_loss = ce_loss_class(iout, label_batch)
                    sum_dice_loss += dice_loss
                    sum_ce_loss += ce_loss
                    loss += (args.dice_weight * dice_loss) + (args.ce_weight * ce_loss)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            
            iter_num += 1
            
            train_dice_loss += sum_dice_loss.item()
            train_ce_loss += sum_ce_loss.item()
            train_loss += loss.item()

            logging.info('epoch %d, iteration %d - dice_loss: %f, ce_loss: %f, loss_total: %f' % (epoch_num, iter_num, sum_dice_loss.item(), sum_ce_loss.item(), loss.item()))
            
            if iter_num % 50 == 0:
                image = image_batch[0, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                
                writer.add_image('train/Image', image, iter_num)
                
                pred = torch.argmax(torch.softmax(P[-1], dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', pred[0, ...] * 50, iter_num)

                labels = label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labels, iter_num)

        train_dice_loss /= len(trainloader)
        train_ce_loss /= len(trainloader)
        train_loss /= len(trainloader)
        
        writer.add_scalar('train/lr', current_lr, epoch_num)
        writer.add_scalar('train/dice_loss', train_dice_loss, epoch_num)
        writer.add_scalar('train/ce_loss', train_ce_loss, epoch_num)
        writer.add_scalar('train/train_loss', train_loss, epoch_num)
        logging.info('Train - epoch %d - train_dice_loss: %f, train_ce_loss: %f, train_loss: %f' % (epoch_num, train_dice_loss, train_ce_loss, train_loss))

        if plateau is not None and plateau.observe(epoch_num, train_loss):
            logging.info('Training-loss plateau at epoch %d; selected epochs: %d', epoch_num, plateau.best_epoch)
            break

    record = {
        'mode': 'training_loss_pilot' if args.training_loss_pilot else 'fixed_epochs',
        'fold_idx': args.fold_idx,
        'training_folds': [k for k in range(5) if k != args.fold_idx],
        'seed': args.seed,
        'configured_max_epochs': args.max_epochs,
        'selected_epochs': plateau.best_epoch if plateau is not None else args.max_epochs,
        'completed_epochs': epoch_num,
        'last_train_loss': train_loss,
    }
    if plateau is not None:
        record['best_train_loss'] = plateau.best_loss
        record['patience'] = args.training_loss_patience
        record['min_delta'] = args.training_loss_min_delta
    else:
        final_model_path = os.path.join(snapshot_path, 'final_model.pth')
        network = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(network.state_dict(), final_model_path)
        logging.info('Final model saved to %s after %d epochs', final_model_path, epoch_num)
    with open(os.path.join(snapshot_path, 'training_record.json'), 'w') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    logging.info('Training completed: %d epochs; selected epochs: %d', epoch_num, record['selected_epochs'])
    writer.close()
    return "Training Finished!"
