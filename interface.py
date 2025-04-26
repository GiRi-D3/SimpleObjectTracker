# replace MyCustomModel with the name of your model
from model import SimpleObjectTracker as TheModel

# change my_descriptively_named_train_function to 
# the function inside train.py that runs the training loop.  
from train import train_SimpleObjectTracker as the_trainer

# change cryptic_inf_f to the function inside predict.py that
# can be called to generate inference on a single image/batch.
from predict import test_SimpleObjectTracker as the_predictor

# change UnicornImgDataset to your custom Dataset class.
from dataset import TrackingNetDataset as TheDataset

# change unicornLoader to your custom dataloader
from dataset import TrackingNetDataLoader as the_dataloader
