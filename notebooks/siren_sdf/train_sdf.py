'''Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement.
'''

# Enable import from parent package
import sys
import os
sys.path.append('siren')

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
from dataclasses import dataclass
import os

@dataclass
class Config:
    logging_root: str = './logs' # root for logging
    experiment_name: str = 'default_experiment' # Name of subdirectory in logging_root where summaries and checkpoints will be saved.
    batch_size: int = 1400
    lr: float = 1e-4 # learning rate. default=5e-5
    num_epochs: int = 10000 # Number of epochs to train for.
    epochs_til_ckpt: int = 1 # Time interval in seconds until checkpoint is saved.
    steps_til_summary: int = 100 # Time interval in seconds until tensorboard summary is saved.
    model_type: str = 'sine' # Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)
    point_cloud_path: str = os.path.expanduser('~/data/point_cloud.xyz')
    checkpoint_path: str = None
    device: str = 'cuda:0'

# Initialize the configuration object
opt = Config(
    experiment_name='siren_sdf_baseline',  # Replace this with your experiment name
    point_cloud_path='/app/notebooks/siren_sdf/data/interior_room.xyz',  # Replace this with your point cloud path
    # Add any other customizations here
)


sdf_dataset = dataio.PointCloud(opt.point_cloud_path, on_surface_points=opt.batch_size)
dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

# Define the model.
if opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=3)
else:
    model = modules.SingleBVPNet(type=opt.model_type, in_features=3)
model.cuda()

# Define the loss
loss_fn = loss_functions.sdf
summary_fn = utils.write_sdf_summary

root_path = os.path.join(opt.logging_root, opt.experiment_name)

training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
               clip_grad=True)
