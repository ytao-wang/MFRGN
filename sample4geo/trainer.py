import os
import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F


def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        # bar = tqdm(dataloader, total=len(dataloader))
        bar = tqdm(dataloader, total=len(dataloader), ncols=150, position=0, leave=True)
    else:
        bar = dataloader
    
    # for loop over one epoch
    for query, reference, ids in bar:
        
        if scaler:
            with autocast():   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                features1, features2 = model(query, reference)
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss = loss_function(features1, features2, model.module.logit_scale.exp())
                else:
                    loss = loss_function(features1, features2, model.logit_scale.exp()) 
                losses.update(loss.item())
                 
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
        else:
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)
            
            # Forward pass
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp()) 
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        
        
        
        if train_config.verbose:
            if len(optimizer.param_groups) > 1:
                monitor = {"loss": "{:.4f}".format(loss.item()),
                        "loss_avg": "{:.4f}".format(losses.avg),
                        "lr1" : "{:.6e}".format(optimizer.param_groups[0]['lr']),
                        "lr2" : "{:.6e}".format(optimizer.param_groups[1]['lr'])}
            else:
                monitor = {"loss": "{:.4f}".format(loss.item()),
                        "loss_avg": "{:.4f}".format(losses.avg),
                        "lr" : "{:.6e}".format(optimizer.param_groups[0]['lr']),}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader, is_autocast=True, input_id=1):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader), ncols=100, position=0, leave=True)
    else:
        bar = dataloader
        
    img_features_list = []
    
    ids_list = []
    with torch.no_grad():
        
        for img, ids in bar:
            
            ids_list.append(ids)
            if is_autocast:
                with autocast():
                    img = img.to(train_config.device)
                    img_feature = model(img, input_id=input_id)
            else:
                img = img.to(train_config.device)
                img_feature = model(img, input_id=input_id)
            
            # normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
      
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return img_features, ids_list


def predict_dual(train_config, model, query_dataloader, reference_dataloader, is_autocast=True):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(query_dataloader, total=len(query_dataloader), ncols=100, position=0, leave=True)
    else:
        bar = query_dataloader
        
    img_features_list1 = []
    img_features_list2 = []
    
    ids_list = []
    reference_iter = iter(reference_dataloader)
    with torch.no_grad():
        
        for query, ids in bar:
            ids_list.append(ids)
            
            if is_autocast:
                with autocast():
                    query = query.to(train_config.device)

                    reference = next(reference_iter)[0]
                    reference = reference.to(train_config.device)
                    img_feature1,  img_feature2 = model(query, reference)
            else:
                query = query.to(train_config.device)
                
                reference = next(reference_iter)[0]
                reference = reference.to(train_config.device)

                img_feature1,  img_feature2= model(query, reference)
            
            # normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature1 = F.normalize(img_feature1, dim=-1)
                img_feature2 = F.normalize(img_feature2, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list1.append(img_feature1.to(torch.float32))
            img_features_list2.append(img_feature2.to(torch.float32))

      
        # keep Features on GPU
        img_features1 = torch.cat(img_features_list1, dim=0)
        img_features2 = torch.cat(img_features_list2, dim=0)

        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return img_features1, img_features2, ids_list, ids_list