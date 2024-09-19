import os
import time
import math
import shutil
import sys
import torch
import pickle
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from sample4geo.dataset.cvact import CVACTDatasetTrain, CVACTDatasetEval, CVACTDatasetTest
from sample4geo.transforms import get_transforms_train, get_transforms_val
from sample4geo.utils import setup_system, Logger
from sample4geo.trainer import train
from sample4geo.evaluate.cvusa_and_cvact import evaluate, calc_sim
from sample4geo.loss import InfoNCE
from model.mfrgn import TimmModel


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num/1000000, 'Trainable': trainable_num/1000000}

@dataclass
class Configuration:
    
    net: str = 'mfrgn_cvact_gpu4_0.0001_e50_autocast'
    net_file = 'model/mfrgn.py'
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k' #'convnext_base.fb_in22k_ft_in1k_384'  convnext_tiny.fb_in22k_ft_in1k_384
    
    # is use polar
    is_polar: bool = False
    psm: bool = True
    image_size_sat = (256, 256)
    img_size_ground = (128, 512)
    
    # Training 
    mixed_precision: bool = True
    seed = 42
    epochs: int = 50
    batch_size: int = 64          # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0,1,2,3)     # GPU ids for training
    
    
    # Similarity Sampling
    custom_sampling: bool = False   # use custom sampling instead of random
    gps_sample: bool = False         # use gps sampling
    sim_sample: bool = False         # use similarity sampling
    neighbour_select: int = 32     # max selection size from pool 
    neighbour_range: int = 64     # pool size for selection
    gps_dict_path: str = "distance_dict/gps_dict_cvact.pkl"   # path to pre-computed distances
 
    # Eval
    batch_size_eval: int = 128
    eval_every_n_epoch: int = 4        # eval every n Epoch
    normalize_features: bool = True

    # Optimizer 
    clip_grad = 100.                   # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False   # Gradient Checkpointing
    
    # Loss
    label_smoothing: float = 0.1
    
    # Learning Rate
    lr: float = 0.0001                # 1 * 10^-4 for ViT | 1 * 10^-3 for CNN
    scheduler: str = "cosine"          # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 1
    lr_end: float = 0.0001             #  only for "polynomial"
    same_lr: bool = True
    
    # Dataset
    data_folder = "/mnt/wangyuntao/Datasets/CVACT_processed"     
    
    # Augment Images
    prob_rotate: float = 0.75          # rotates the sat image and ground images simultaneously
    prob_flip: float = 0.5             # flipping the sat image and ground images simultaneously
    
    # Savepath for model checkpoints 
    model_path: str = "results"
    
    # Eval before training
    zero_shot: bool = False
    
    # Checkpoint to start from
    checkpoint_start = None
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 8
    
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

    torch.backends.cudnn.enabled = False

    s1 = time.time()
    model_path = "{}/{}/{}_{}".format(config.model_path,
                                   config.model,
                                   config.net,
                                   time.strftime("%m-%d-%H-%M-%S"))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))
    shutil.copyfile(config.net_file, "{}/model.py".format(model_path))

    # Redirect print to both console and log file
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
                          
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    image_size_sat = config.image_size_sat
    img_size_ground = config.img_size_ground
    
    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
     
    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     
    
    params = get_parameter_number(model)
    print(f"Total: {params['Total']} M   Trainable: {params['Trainable']} M") 

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
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
                                                                   is_polar=config.is_polar
                                                                   )
                                                                   
                                                                   
    # Train
    train_dataset = CVACTDatasetTrain(data_folder=config.data_folder ,
                                      transforms_query=ground_transforms_train,
                                      transforms_reference=sat_transforms_train,
                                      prob_flip=config.prob_flip,
                                      prob_rotate=config.prob_rotate,
                                      shuffle_batch_size=config.batch_size,
                                      is_polar=config.is_polar
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
                                                               is_polar=config.is_polar
                                                               )


    # Reference Satellite Images
    reference_dataset_val = CVACTDatasetEval(data_folder=config.data_folder ,
                                             split="val",
                                             img_type="reference",
                                             transforms=sat_transforms_val,
                                             is_polar=config.is_polar
                                             )
    
    reference_dataloader_val = DataLoader(reference_dataset_val,
                                          batch_size=config.batch_size_eval,
                                          num_workers=config.num_workers,
                                          shuffle=False,
                                          pin_memory=True)
    
    
    
    # Query Ground Images Test
    query_dataset_val = CVACTDatasetEval(data_folder=config.data_folder ,
                                         split="val",
                                         img_type="query",    
                                         transforms=ground_transforms_val,
                                         is_polar=config.is_polar
                                         )
    
    query_dataloader_val = DataLoader(query_dataset_val,
                                      batch_size=config.batch_size_eval,
                                      num_workers=config.num_workers,
                                      shuffle=False,
                                      pin_memory=True)
    
    
    print("Reference Images Val:", len(reference_dataset_val))
    print("Query Images Val:", len(query_dataset_val))
    
    
    #-----------------------------------------------------------------------------#
    # GPS Sample                                                                  #
    #-----------------------------------------------------------------------------#
    if config.gps_sample:
        with open(config.gps_dict_path, "rb") as f:
            sim_dict = pickle.load(f)
    else:
        sim_dict = None

    #-----------------------------------------------------------------------------#
    # Sim Sample                                                                  #
    #-----------------------------------------------------------------------------#
    
    if config.sim_sample:
    
        # Query Ground Images Train for simsampling
        query_dataset_train = CVACTDatasetEval(data_folder=config.data_folder ,
                                               split="train",
                                               img_type="query",   
                                               transforms=ground_transforms_val,
                                               is_polar=config.is_polar
                                               )
            
        query_dataloader_train = DataLoader(query_dataset_train,
                                            batch_size=config.batch_size_eval,
                                            num_workers=config.num_workers,
                                            shuffle=False,
                                            pin_memory=True)
        
        
        reference_dataset_train = CVACTDatasetEval(data_folder=config.data_folder ,
                                                   split="train",
                                                   img_type="reference", 
                                                   transforms=sat_transforms_val,
                                                   is_polar=config.is_polar
                                                   )
        
        reference_dataloader_train = DataLoader(reference_dataset_train,
                                                batch_size=config.batch_size_eval,
                                                num_workers=config.num_workers,
                                                shuffle=False,
                                                pin_memory=True)


        print("\nReference Images Train:", len(reference_dataset_train))
        print("Query Images Train:", len(query_dataset_train))        

    
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
                # for bk in [model.module.backbone_sat, model.module.backbone_grd]:
                #     ignored_params += list(map(id, bk.parameters()))
            else:
                ignored_params += list(map(id, model.backbone.parameters()))
                # for bk in [model.backbone_sat, model.backbone_grd]:
                #     ignored_params += list(map(id, bk.parameters()))
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
                           reference_dataloader=reference_dataloader_val,
                           query_dataloader=query_dataloader_val, 
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
    start_epoch = 0   
    best_score = 0
    r1_test = 0
    

    for epoch in range(1, config.epochs+1):
        
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
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) and epoch != config.epochs:
        
            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))
        
            r1_test = evaluate(config=config,
                               model=model,
                               reference_dataloader=reference_dataloader_val,
                               query_dataloader=query_dataloader_val, 
                               ranks=[1, 5, 10],
                               step_size=1000,
                               is_dual=True,
                               is_autocast=False,
                               cleanup=True)
            
            if config.sim_sample:
                r1_train, sim_dict = calc_sim(config=config,
                                              model=model,
                                              reference_dataloader=reference_dataloader_train,
                                              query_dataloader=query_dataloader_train, 
                                              ranks=[1, 5, 10],
                                              step_size=1000,
                                              is_dual=True,
                                              is_autocast=True,
                                              cleanup=True)
            s5 = time.time()

            if r1_test > best_score:

                best_score = r1_test

                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                

        if config.custom_sampling:
            train_dataloader.dataset.shuffle(sim_dict,
                                             neighbour_select=config.neighbour_select,
                                             neighbour_range=config.neighbour_range)
                
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))  


    # #-----------------------------------------------------------------------------#
    # # Test                                                                        #
    # #-----------------------------------------------------------------------------#
    
    # # Reference Satellite Images
    # reference_dataset_test = CVACTDatasetTest(data_folder=config.data_folder ,
    #                                           img_type="reference",
    #                                           transforms=sat_transforms_val)
    
    # reference_dataloader_test = DataLoader(reference_dataset_test,
    #                                        batch_size=config.batch_size_eval,
    #                                        num_workers=config.num_workers,
    #                                        shuffle=False,
    #                                        pin_memory=True)
    
    
    
    # # Query Ground Images Test
    # query_dataset_test = CVACTDatasetTest(data_folder=config.data_folder ,
    #                                       img_type="query",    
    #                                       transforms=ground_transforms_val,
    #                                       )
    
    # query_dataloader_test = DataLoader(query_dataset_test,
    #                                    batch_size=config.batch_size_eval,
    #                                    num_workers=config.num_workers,
    #                                    shuffle=False,
    #                                    pin_memory=True)
    
    
    # print("Reference Images Test:", len(reference_dataset_test))
    # print("Query Images Test:", len(query_dataset_test))          


    # print("\n{}[{}]{}".format(30*"-", "Test", 30*"-"))  

  
    # r1_test = evaluate(config=config,
    #                    model=model,
    #                    reference_dataloader=reference_dataloader_test,
    #                    query_dataloader=query_dataloader_test, 
    #                    ranks=[1, 5, 10],
    #                    step_size=1000,
    #                    cleanup=True)
        
    # time.sleep(10)
    # cmd = 'sh moni1.sh'
    # os.system(cmd)
