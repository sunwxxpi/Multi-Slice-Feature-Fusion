import os
import sys
import random
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm
from utils import PolyLRScheduler, DiceLoss
from datasets.dataset import shuffle_within_batch, COCA_dataset, RandomAugmentation, Resize, ToTensor

def trainer_coca(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", 
                        level=logging.INFO, 
                        format='[%(asctime)s.%(msecs)03d] %(message)s', 
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    
    train_transform = T.Compose([
            RandomAugmentation(),
            Resize(output_size=[args.img_size, args.img_size]),
            ToTensor()
        ])
    db_train = COCA_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        transform=train_transform
    )
    
    val_transform = T.Compose([
        Resize(output_size=[args.img_size, args.img_size]),
        ToTensor()
    ])
    db_val = COCA_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="val",
        transform=val_transform
    )

    print("The length of train set is: {}".format(len(db_train)))
    print("The length of validation set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=False, 
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn, 
                             collate_fn=shuffle_within_batch)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.train()
    
    dice_loss_class = DiceLoss(num_classes)
    ce_loss_class = CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    
    max_iterations = args.max_epochs * len(trainloader)
    scheduler = PolyLRScheduler(optimizer, initial_lr=base_lr, max_steps=max_iterations)
    
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    iter_num = 0
    max_epoch = args.max_epochs
    best_val_loss = float('inf')
    best_model_path = None
    
    for epoch_num in tqdm(range(1, max_epoch + 1), ncols=70):
        train_dice_loss = 0.0
        train_ce_loss = 0.0
        train_loss = 0.0
        
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader, start=1):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = model(image_batch)
            
            dice_loss = dice_loss_class(outputs, label_batch, softmax=True)
            ce_loss = ce_loss_class(outputs, label_batch)
            loss = (0.5 * dice_loss) + (0.5 * ce_loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            
            iter_num += 1
            
            train_dice_loss += dice_loss.item()
            train_ce_loss += ce_loss.item()
            train_loss += loss.item()

            logging.info('epoch %d, iteration %d - dice_loss: %f, ce_loss: %f, loss_total: %f' % (epoch_num, iter_num, dice_loss.item(), ce_loss.item(), loss.item()))
            
            if iter_num % 50 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                
                writer.add_image('train/Image', image, iter_num)
                
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)

                labels = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labels, iter_num)

        train_dice_loss /= len(trainloader)
        train_ce_loss /= len(trainloader)
        train_loss /= len(trainloader)
        
        writer.add_scalar('train/lr', current_lr, epoch_num)
        writer.add_scalar('train/dice_loss', train_dice_loss, epoch_num)
        writer.add_scalar('train/ce_loss', train_ce_loss, epoch_num)
        writer.add_scalar('train/train_loss', train_loss, epoch_num)
        logging.info('Train - epoch %d - train_dice_loss: %f, train_ce_loss: %f, train_loss: %f' % (epoch_num, train_dice_loss, train_ce_loss, train_loss))

        # Validation step after each epoch
        val_dice_loss = 0.0
        val_ce_loss = 0.0
        val_loss = 0.0
        
        model.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader, start=1):
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                
                outputs = model(image_batch)
                
                dice_loss = dice_loss_class(outputs, label_batch, softmax=True)
                ce_loss = ce_loss_class(outputs, label_batch)
                loss = (0.5 * dice_loss) + (0.5 * ce_loss)
                
                val_dice_loss += dice_loss.item()
                val_ce_loss += ce_loss.item()
                val_loss += loss.item()

        # Epoch별 평균 validation 손실 계산 및 기록
        val_dice_loss /= len(valloader)
        val_ce_loss /= len(valloader)
        val_loss /= len(valloader)

        writer.add_scalar('val/dice_loss', val_dice_loss, epoch_num)
        writer.add_scalar('val/ce_loss', val_ce_loss, epoch_num)
        writer.add_scalar('val/val_loss', val_loss, epoch_num)
        logging.info('Validation - epoch %d - val_dice_loss: %f, val_ce_loss: %f, val_loss: %f' % (epoch_num, val_dice_loss, val_ce_loss, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            
            best_model_path = os.path.join(snapshot_path, f'epoch_{epoch_num}_{best_val_loss:.4f}_best_model.pth')
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), best_model_path)
            else:
                torch.save(model.state_dict(), best_model_path)
                
            logging.info(f"Best model saved to {best_model_path} with val_loss: {best_val_loss:.4f}")

        if epoch_num == max_epoch:
            save_model_path = os.path.join(snapshot_path, f'epoch_{epoch_num}_{val_loss:.4f}.pth')
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_model_path)
            else:
                torch.save(model.state_dict(), save_model_path)
                
            logging.info(f"Final epoch model saved to {save_model_path} with val_loss: {val_loss:.4f}")

    writer.close()
    return "Training Finished!"