import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from sample4geo.dataset.vigor import VigorDatasetEval
from sample4geo.transforms import get_transforms_val
from sample4geo.evaluate.vigor import evaluate
# from sample4geo.model import TimmModel
from model.mfrgn import TimmModel

@dataclass
class Configuration:
    
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k'
    
    # Override model image size
    is_polar: bool = False

    image_size_sat = (320, 320)
    img_size_ground = (320, 640)
    
    # Evaluation 
    batch_size: int = 64
    verbose: bool = True
    gpu_ids: tuple = (0,1,2,3)
    normalize_features: bool = True    
    
    # Dataset
    data_folder = "/mnt/wangyuntao/Datasets/VIGOR_processed_320"
    same_area: bool = False            # True: same | False: cross
    ground_cutting = 0                 # cut ground upper and lower
   
    # Checkpoint to start from
    checkpoint_start = '/mnt/wangyuntao/ACMMM2024-results/weights/weights_vigor_cross.pth' 
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 8 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    

#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration() 


if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))


    model = TimmModel(config.model,
                      config.image_size_sat,
                      config.img_size_ground,
                      psm=True,
                      is_polar=config.is_polar)
                          
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    image_size_sat = config.image_size_sat
    img_size_ground = config.img_size_ground
     
    # load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     

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


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#
    
    # Eval
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std,
                                                                   is_polar=config.is_polar,
                                                                   ground_cutting=config.ground_cutting)


    # Reference Satellite Images Test
    reference_dataset_test = VigorDatasetEval(data_folder=config.data_folder ,
                                              split="test",
                                              img_type="reference",
                                              same_area=config.same_area,  
                                              transforms=sat_transforms_val,
                                              )
    
    reference_dataloader_test = DataLoader(reference_dataset_test,
                                           batch_size=config.batch_size,
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
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    
    print("Query Images Test:", len(query_dataset_test))
    print("Reference Images Test:", len(reference_dataset_test))
    
    #-----------------------------------------------------------------------------#
    # Evaluate                                                                    #
    #-----------------------------------------------------------------------------#

    print("\n{}[{}]{}".format(30*"-", "VIGOR Cross", 30*"-"))  

    r1_test = evaluate(config=config,
                       model=model,
                       reference_dataloader=reference_dataloader_test,
                       query_dataloader=query_dataloader_test, 
                       ranks=[1, 5, 10],
                       step_size=1000,
                       is_dual=False,
                       is_autocast=True,
                       cleanup=True)
 
