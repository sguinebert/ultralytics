import sys
sys.path.append('/home/guinebert/repos/yolov8_/')  # Replace with the actual path to the cloned repository directory
print(sys.path)

from ultralytics.models import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model 

# Train the model
results = model.train(data="/media/guinebert/data/MURA/MURA.yaml", epochs=3)
