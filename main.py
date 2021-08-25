import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

from datasets import list_dataset
from core.optimization import *
from cfg import parser
from core.utils import *
from core.region_loss import RegionLoss
from core.model import YOWO, get_fine_tuning_parameters

if __name__ == '__main__':
    # load configuration arguments
    args  = parser.parse_args()
    cfg   = parser.load_config(args)

    # check backup directory, create if necessary
    if not os.path.exists(cfg.BACKUP_DIR):
        os.makedirs(cfg.BACKUP_DIR)

    # create model
    model = YOWO(cfg)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None) # in multi-gpu case
    # print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging('Total number of trainable parameters: {}'.format(pytorch_total_params))

    seed = int(time.time())
    torch.manual_seed(seed)
    use_cuda = True
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' # TODO: add to config e.g. 0,1,2,3
        torch.cuda.manual_seed(seed)

    # create optimizer
    parameters = get_fine_tuning_parameters(model, cfg)
    optimizer = torch.optim.Adam(parameters, lr=cfg.TRAIN.LEARNING_RATE, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    best_score   = 0 # initialize best score


    # # load resume path if necessary
    # if cfg.TRAIN.RESUME_PATH:
    #     print('loading checkpoint {}'.format(cfg.TRAIN.RESUME_PATH))
    #     checkpoint = torch.load(cfg.TRAIN.RESUME_PATH)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     del checkpoint


    # Data loader, training scheme and loss function are different for AVA and UCF24/JHMDB21 datasets
    dataset = cfg.TRAIN.DATASET
    assert dataset == 'ucsp', 'invalid dataset'

    if dataset == 'ucsp':
        train_dataset = list_dataset.UCSP_Dataset(cfg.LISTDATA.BASE_PTH, cfg.LISTDATA.TRAIN_FILE, dataset=dataset,
                        shape=(cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.TRAIN_CROP_SIZE),
                        transform=transforms.Compose([transforms.ToTensor()]), 
                        train=True, clip_duration=cfg.DATA.NUM_FRAMES, sampling_rate=cfg.DATA.SAMPLING_RATE)
        print("##### len(train_dataset): ", len(train_dataset)) # trainlist.txt items amount "Basketball/v_Basketball_g15_c05/00064.txt"
        
        train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size= cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                                num_workers=cfg.DATA_LOADER.NUM_WORKERS, drop_last=True, pin_memory=True)
        print("##### len(train_loader): ", len(train_loader))
    
        test_dataset  = list_dataset.UCSP_Dataset(cfg.LISTDATA.BASE_PTH, cfg.LISTDATA.TEST_FILE, dataset=dataset,
                        shape=(cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.TRAIN_CROP_SIZE),
                        transform=transforms.Compose([transforms.ToTensor()]), 
                        train=False, clip_duration=cfg.DATA.NUM_FRAMES, sampling_rate=cfg.DATA.SAMPLING_RATE)
        print("##### len(test_dataset): ", len(test_dataset))
    
        test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size= cfg.TRAIN.BATCH_SIZE, shuffle=False,
                                                num_workers=cfg.DATA_LOADER.NUM_WORKERS, drop_last=False, pin_memory=True)
        print("##### len(test_loader): ", len(test_loader))
        
        loss_module   = RegionLoss(cfg).cuda()

        train = getattr(sys.modules[__name__], 'train_ucsp')
        test  = getattr(sys.modules[__name__], 'test_ucsp')

    # Training and Testing Schedule
    if cfg.TRAIN.EVALUATE:
        logging('evaluating***********************')
        test(cfg, 0, model, test_loader)
    else:
        logging('training**************************')
        for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH + 1):
            lr_new = adjust_learning_rate(optimizer, epoch, cfg) # Adjust learning rate
            logging('training at epoch %d, lr %f' % (epoch, lr_new)) # Train and test model
            train(cfg, epoch, model, train_loader, loss_module, optimizer) 
            logging('testing at epoch %d' % (epoch))
            score = test(cfg, epoch, model, test_loader)

            # Save the model to backup directory
            is_best = score > best_score
            if is_best:
                print("New best score is achieved: ", score)
                print("Previous score was: ", best_score)
                best_score = score

            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'score': score
                }
            save_checkpoint(state, is_best, cfg.BACKUP_DIR, cfg.TRAIN.DATASET, cfg.DATA.NUM_FRAMES)
            logging('Weights are saved to backup directory: %s' % (cfg.BACKUP_DIR))