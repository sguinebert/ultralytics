import sys
sys.path.append('/home/guinebert/repos/yolov8_/')  # Replace with the actual path to the cloned repository directory
print(sys.path)

from ultralytics.models import YOLO
from ultralytics import RTDETR

# Load a yolov8 model
#yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt	
#yolov8n-seg.pt, yolov8s-seg.pt, yolov8m-seg.pt, yolov8l-seg.pt, yolov8x-seg.pt	
#yolov8n-pose.pt, yolov8s-pose.pt, yolov8m-pose.pt, yolov8l-pose.pt, yolov8x-pose.pt, yolov8x-pose-p6.pt
#yolov8n-cls.pt, yolov8s-cls.pt, yolov8m-cls.pt, yolov8l-cls.pt, yolov8x-cls.pt	
model = YOLO('yolov8n.yaml') #.load('yolov8x.pt')  # build from YAML and transfer weights

# Load a COCO-pretrained RT-DETR-l model
#rtdetr-l.pt rtdetr-x.pt
#model = RTDETR('rtdetr-l.pt')

# Display model information (optional)
#model.info()

# Train the model
results =   model.train(#data="coco128.yaml",
                        data="/media/guinebert/data/MURA/MURA.yaml", 
                        #resume=True,
                        val=True,            # validate/test during training
                        epochs=3,
                        imgsz=640,
                        batch=16,
                        optimizer='auto', # optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
                        device=[0,1,2,3], # device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
                        freeze=None,       # (int or list, optional) freeze first n layers, or freeze list of layer indices during training
                        
                        #seed=0,           # random seed for reproducibility
                        #deterministic=True, # whether to enable deterministic mode
                        amp=True,          # Automatic Mixed Precision (AMP) training, choices=[True, False]
                        profile=False,     # profile ONNX and TensorRT speeds during training for loggers
                        dropout=0.0,        # use dropout regularization (classify train only)
                        
                        
                        single_cls=False,  # train multi-class data as single-class
                        rect=False,        # rectangular training with each batch collated for minimum padding
                        cos_lr=False,      # use cosine learning rate scheduler
                        close_mosaic=10,   # (int) disable mosaic augmentation for final epochs (0 to disable)
                        fraction=1.0,      # dataset fraction to train on (default is 1.0, all images in train set)

                        cache=False,      # True/ram, disk or False. Use cache for data loading
                                
                        patience=50,      # epochs to wait for no observable improvement for early stopping of training
                        lr0=0.01,          # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
                        lrf=0.01,          # final learning rate (lr0 * lrf)
                        momentum=0.937,    # SGD momentum/Adam beta1
                        weight_decay=0.0005, # optimizer weight decay 5e-4
                        warmup_epochs=3.0, # warmup epochs (fractions ok)
                        warmup_momentum=0.8, # warmup initial momentum
                        warmup_bias_lr=0.1,  # warmup initial bias lr
                        box=1.0,            # box loss gain
                        cls=20,            # cls loss gain (scale with pixels)
                        dfl=1.5,            # dfl loss gain
                        #pose=12.0,          # pose loss gain (pose-only)
                        #kobj=2.0,           # keypoint obj loss gain (pose-only)
                        #workers=8,        # number of worker threads for data loading (per RANK if DDP)
                        save=True) # save train checkpoints and predict results
                        

    

   
    # verbose=False,    # whether to print verbose output
    
    #save_period=-1,   # Save checkpoint every x epochs (disabled if < 1)
    # project=None,     # project name
    # name=None,        # experiment name
    # exist_ok=False,   # whether to overwrite existing experiment
    # pretrained=False, # whether to use a pretrained model
    
    # label_smoothing=0.0, # label smoothing (fraction)
    # nbs=64,             # nominal batch size
    # overlap_mask=True,  # masks should overlap during training (segment train only)
    # mask_ratio=4,       # mask downsample ratio (segment train only)