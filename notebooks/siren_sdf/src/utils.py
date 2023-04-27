import torch
import numpy as np
import matplotlib.pyplot as plt

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords

def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)

import numpy as np
from pathlib import Path
import io
import matplotlib.pyplot as plt
from PIL import Image

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    pil_image = Image.open(buf)
    return pil_image

def make_contour_plot(array_2d, mode='log', title=''):
    fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)

    if(mode=='log'):
        num_levels = 6
        levels_pos = np.logspace(-2, 0, num=num_levels) # logspace
        levels_neg = -1. * levels_pos[::-1]
        levels = np.concatenate((levels_neg, np.zeros((0)), levels_pos), axis=0)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels*2+1))
    elif(mode=='lin'):
        num_levels = 10
        levels = np.linspace(-.5,.5,num=num_levels)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels))

    sample = np.flipud(array_2d)
    CS = ax.contourf(sample, levels=levels, colors=colors)
    cbar = fig.colorbar(CS)
    
    if title != '':
        fig.title = title
    
    ax.contour(sample, levels=levels, colors='k', linewidths=0.1)
    ax.contour(sample, levels=[0], colors='k', linewidths=0.3)
    ax.axis('off')
    pil_img = fig_to_pil(fig)
    plt.close('all')
    return pil_img

def get_sdf_summary(model, model_input, gt, model_output):
    slice_coords_2d = get_mgrid(512)
    result = {}

    with torch.no_grad():
        yz_slice_coords = torch.cat((torch.zeros_like(slice_coords_2d[:, :1]), slice_coords_2d), dim=-1)
        yz_slice_model_input = {'coords': yz_slice_coords.to(model_output['model_out'].device)[None, ...]}

        yz_model_out = model(yz_slice_model_input)
        sdf_values = yz_model_out['model_out']
        sdf_values = lin2img(sdf_values).squeeze().cpu().numpy()
        fig = make_contour_plot(sdf_values, title='yz_sdf_slice')
        
        result['yz_sdf_slice'] = fig
        
        # writer.add_figure(prefix + 'yz_sdf_slice', fig, global_step=total_steps)

        xz_slice_coords = torch.cat((slice_coords_2d[:,:1],
                                     torch.zeros_like(slice_coords_2d[:, :1]),
                                     slice_coords_2d[:,-1:]), dim=-1)
        xz_slice_model_input = {'coords': xz_slice_coords.to(model_output['model_out'].device)[None, ...]}

        xz_model_out = model(xz_slice_model_input)
        sdf_values = xz_model_out['model_out']
        sdf_values = lin2img(sdf_values).squeeze().cpu().numpy()
        fig = make_contour_plot(sdf_values, title='xz_sdf_slice')
        
        result['xz_sdf_slice'] = fig
        # writer.add_figure(prefix + 'xz_sdf_slice', fig, global_step=total_steps)

        xy_slice_coords = torch.cat((slice_coords_2d[:,:2],
                                     -0.75*torch.ones_like(slice_coords_2d[:, :1])), dim=-1)
        xy_slice_model_input = {'coords': xy_slice_coords.to(model_output['model_out'].device)[None, ...]}

        xy_model_out = model(xy_slice_model_input)
        sdf_values = xy_model_out['model_out']
        sdf_values = lin2img(sdf_values).squeeze().cpu().numpy()
        fig = make_contour_plot(sdf_values, title='xy_sdf_slice')
        
        result['xy_sdf_slice'] = fig
        # writer.add_figure(prefix + 'xy_sdf_slice', fig, global_step=total_steps)

        result['model_out_min_max'] = model_output['model_out'].min().detach().cpu().numpy()
        result['coords'] = model_input['coords'].min().detach().cpu().numpy()
        
        return result