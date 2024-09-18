import os
import time
import math
import shutil
import sys
import torch
import gc
import pickle
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from sample4geo.dataset.vigor import VigorDatasetEval, VigorDatasetTrain
from sample4geo.transforms import get_transforms_train, get_transforms_val
from sample4geo.utils import setup_system, Logger
from sample4geo.trainer import train
from sample4geo.evaluate.vigor import evaluate, calc_sim
from sample4geo.loss import InfoNCE
from model.mfrgn import TimmModel

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num/1000000, 'Trainable': trainable_num/1000000}


@dataclass
class Configuration:
    
    net: str = 'vigor_'
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k'

    is_save_all_model: bool = False
    
    # is use polar
    img_size = 320
    is_polar: bool = False
    image_size_sat = (img_size, img_size)
    img_size_ground = (img_size, img_size*2)
    psm: bool = True
    
    # Training 
    mixed_precision: bool = True
    seed = 40
    epochs: int = 70
    batch_size: int = 64        # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0,1,2,3)   # GPU ids for training
    
    
    # Similarity Sampling
    custom_sampling: bool = False   # use custom sampling instead of random
    gps_sample: bool = False        # use gps sampling
    sim_sample: bool = False        # use similarity sampling
    neighbour_select: int = 64     # max selection size from pool
    neighbour_range: int = 128     # pool size for selection
    gps_dict_path: str = "distance_dict/gps_dict_cross.pkl"   # gps_dict_cross.pkl | gps_dict_same.pkl
 
    # Eval
    batch_size_eval: int = 128
    eval_every_n_epoch: int = 1      # eval every n Epoch
    normalize_features: bool = True

    # Optimizer 
    clip_grad = 100.                 # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False # Gradient Checkpointing
    
    # Loss
    label_smoothing: float = 0.1
    
    # Learning Rate
    lr: float = 0.0001                  # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = "cosine"          # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 1
    lr_end: float = 0.0001             #  only for "polynomial"
    
    # Dataset
    data_folder = "/mnt/wangyuntao/Datasets/VIGOR_processed_320"
    same_area: bool = False             # True: same | False: cross
    ground_cutting = 0                 # cut ground upper and lower
    same_lr: bool = True
   
    # Augment Images
    prob_rotate: float = 0.75          # rotates the sat image and ground images simultaneously
    prob_flip: float = 0.5             # flipping the sat image and ground images simultaneously
    
    # Savepath for model checkpoints
    model_path: str = "/mnt/wangyuntao/Sample4Geo-results/vigor"
    
    # Eval before training
    zero_shot: bool = False  
    
    # Checkpoint to start from
    resume: bool = False
    checkpoint_start = '' 
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 16
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    # # for better performance
    # cudnn_benchmark: bool = True
    # # make cudnn deterministic
    # cudnn_deterministic: bool = False

    # 选择默认卷积算法， 模型输出结果不变
    cudnn_benchmark: bool = False
    # make cudnn deterministic
    cudnn_deterministic: bool = True



#-----------------------------------------------------------------------------#
# Train Config                                                                #
#-----------------------------------------------------------------------------#

config = Configuration() 


if __name__ == '__main__':

    s1 = time.time()
    model_path = "{}/{}/{}_{}".format(config.model_path,
                                   config.model,
                                   config.net,
                                   time.strftime("%m-%d-%H-%M-%S"))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    # Redirect print to both console and log file
    if config.resume and config.checkpoint_start is not None:
        model_path = config.checkpoint_start.split('/weights')[0]
        sys.stdout = Logger(os.path.join(model_path, 'log_resume.txt'))
    else:
        sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))

    model = TimmModel(config.model,
                      config.image_size_sat,
                      config.img_size_ground,
                      psm=config.psm,
                      is_polar=config.is_polar)
                          
    data_config = model.get_config()
    print(data_config)

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    
    image_size_sat = config.image_size_sat
    img_size_ground = config.img_size_ground
    
    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
     
    # Load pretrained Checkpoint    
    if config.resume and  config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        checkpoint = torch.load(config.checkpoint_start)  
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)     

    params = get_parameter_number(model)
    print(f"Total: {params['Total']} M   Trainable: {params['Trainable']} M")

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        # model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        model = torch.nn.DataParallel(model)
            
    # Model to device   
    model = model.to(config.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))
    print("lr: ", config.lr)
    print("batch size: ", config.batch_size)


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    sat_transforms_train, ground_transforms_train = get_transforms_train(image_size_sat,
                                                                         img_size_ground,
                                                                         mean=mean,
                                                                         std=std,
                                                                         ground_cutting=config.ground_cutting,
                                                                         is_polar=config.is_polar)
                                                                   
                                                                   
    # Train
    train_dataset = VigorDatasetTrain(data_folder=config.data_folder ,
                                      same_area=config.same_area,
                                      transforms_query=ground_transforms_train,
                                      transforms_reference=sat_transforms_train,
                                      prob_flip=config.prob_flip,
                                      prob_rotate=config.prob_rotate,
                                      shuffle_batch_size=config.batch_size
                                      )
    
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)
    
    
    # Eval
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std,
                                                                   ground_cutting=config.ground_cutting,
                                                                   is_polar=config.is_polar)


    # Reference Satellite Images Test
    reference_dataset_test = VigorDatasetEval(data_folder=config.data_folder ,
                                              split="test",
                                              img_type="reference",
                                              same_area=config.same_area,  
                                              transforms=sat_transforms_val,
                                              )
    
    reference_dataloader_test = DataLoader(reference_dataset_test,
                                           batch_size=config.batch_size_eval,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)
    
    
    
    # Query Ground Images Test
    query_dataset_test = VigorDatasetEval(data_folder=config.data_folder ,
                                          split="test",
                                          img_type="query",
                                          same_area=config.same_area,      
                                          transforms=ground_transforms_val,
                                          )
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    
    print("Query Images Test:", len(query_dataset_test))
    print("Reference Images Test:", len(reference_dataset_test))
    

    #-----------------------------------------------------------------------------#
    # GPS Sample                                                                  #
    #-----------------------------------------------------------------------------#
    if config.gps_sample:
        with open(config.gps_dict_path, "rb") as f:
            sim_dict = pickle.load(f)
    else:
        sim_dict = None
    
    #-----------------------------------------------------------------------------#
    # Sim Sample + Eval on Train                                                  #
    #-----------------------------------------------------------------------------#
    
    if config.sim_sample:

        # Query Ground Images Train for simsampling
        query_dataset_train = VigorDatasetEval(data_folder=config.data_folder ,
                                               split="train",
                                               img_type="query",
                                               same_area=config.same_area,      
                                               transforms=ground_transforms_val,
                                               )
            
        query_dataloader_train = DataLoader(query_dataset_train,
                                            batch_size=config.batch_size_eval,
                                            num_workers=config.num_workers,
                                            shuffle=False,
                                            pin_memory=True)
        
        # Reference Satellite Images Train for simsampling
        reference_dataset_train = VigorDatasetEval(data_folder=config.data_folder ,
                                                   split="train",
                                                   img_type="reference",
                                                   same_area=config.same_area,  
                                                   transforms=sat_transforms_val,
                                                   )
        
        reference_dataloader_train = DataLoader(reference_dataset_train,
                                                batch_size=config.batch_size_eval,
                                                num_workers=config.num_workers,
                                                shuffle=False,
                                                pin_memory=True)
            
      
        print("\nQuery Images Train:", len(query_dataset_train))
        print("Reference Images Train (unique):", len(reference_dataset_train))
        
    
    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = InfoNCE(loss_function=loss_fn,
                            device=config.device,
                            )

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None
        
    #-----------------------------------------------------------------------------#
    # optimizer                                                                   #
    #-----------------------------------------------------------------------------#

    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        if config.same_lr:
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        else:
            ignored_params = []
            if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                ignored_params += list(map(id, model.module.backbone.parameters()))
            else:
                ignored_params += list(map(id, model.backbone.parameters()))
            extra_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
            base_params = filter(lambda p: id(p) in ignored_params, model.parameters())
            optimizer = torch.optim.AdamW([
            {'params': base_params, 'lr': 0.7 * config.lr},
            {'params': extra_params, 'lr': config.lr}
        ])


    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs
       
    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))  
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end = config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))   
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))   
        scheduler =  get_constant_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=warmup_steps)
           
    else:
        scheduler = None
        
    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))
        
        
    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))  

        r1_test = evaluate(config=config,
                           model=model,
                           reference_dataloader=reference_dataloader_test,
                           query_dataloader=query_dataloader_test, 
                           ranks=[1, 5, 10],
                           step_size=1000,
                           is_dual=True,
                           cleanup=True)
        
        if config.sim_sample:
            r1_train, sim_dict = calc_sim(config=config,
                                          model=model,
                                          reference_dataloader=reference_dataloader_train,
                                          query_dataloader=query_dataloader_train, 
                                          ranks=[1, 5, 10],
                                          step_size=1000,
                                          is_dual=True,
                                          cleanup=True)
        
    #-----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    #-----------------------------------------------------------------------------#            
    if config.custom_sampling:
        train_dataloader.dataset.shuffle(sim_dict,
                                         neighbour_select=config.neighbour_select,
                                         neighbour_range=config.neighbour_range)
            
    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    

    start_epoch = 1   
    best_score = 0
    r1_test = 0

    if config.resume:
        start_epoch = checkpoint['epoch']

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print('-------resume start from epoch {}-------'.format(start_epoch))
    

    for epoch in range(start_epoch, config.epochs+1):
        
        print("\n{}[Epoch: {}/{}]{}".format(30*"-", epoch, config.epochs, 30*"-"))
        s2 = time.time()

        train_loss = train(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler)
        
        if len(optimizer.param_groups) > 1:
            print("Epoch: {}, Train Loss = {:.3f}, Lr1 = {:.6e}, Lr2 = {:.6e}".format(epoch,
                                                                    train_loss,
                                                                    optimizer.param_groups[0]['lr'],
                                                                    optimizer.param_groups[1]['lr']))
        else:
            print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6e}".format(epoch,
                                                                    train_loss,
                                                                    optimizer.param_groups[0]['lr']))
        s3 = time.time()
        print('train: ', (s3 - s2)/60, 'min')
               

        # evaluate
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
        
            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))
            # 由于硬件限制，边训练边评估会极大增加训练时间，可以训练完之后进行评估。
            # if epoch >= 36:
            #     s4 = time.time()
            #     r1_test = evaluate(config=config,
            #                     model=model,
            #                     reference_dataloader=reference_dataloader_test,
            #                     query_dataloader=query_dataloader_test, 
            #                     ranks=[1, 5, 10],
            #                     step_size=1000,
            #                     cleanup=True,
            #                     is_dual=False,
            #                     is_autocast=True)
            #     s5 = time.time()
            #     print('val: ', (s5 - s4)/60, 'min')

            if config.sim_sample and epoch != config.epochs:
                s6 = time.time()
                r1_train, sim_dict = calc_sim(config=config,
                                              model=model,
                                              reference_dataloader=reference_dataloader_train,
                                              query_dataloader=query_dataloader_train, 
                                              ranks=[1, 5, 10],
                                              step_size=1000,
                                              cleanup=True,
                                              is_dual=False,
                                              is_autocast=True)
                
                s7 = time.time()
                print('sim_sample: ', (s7 - s6)/60, 'min')
                
            if r1_test > best_score:

                best_score = r1_test

                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}_b.pth'.format(model_path, epoch, r1_test))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}_b.pth'.format(model_path, epoch, r1_test))

            else:
                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
        else:
            if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                torch.save(model.module.state_dict(), '{}/weights_e{}.pth'.format(model_path, epoch))
            else:
                torch.save(model.state_dict(), '{}/weights_e{}.pth'.format(model_path, epoch))
                

        if config.custom_sampling:
            train_dataloader.dataset.shuffle(sim_dict,
                                             neighbour_select=config.neighbour_select,
                                             neighbour_range=config.neighbour_range)
            
        gc.collect()
                
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))
    gc.collect()


