import lovely_tensors as lt

lt.monkey_patch()

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt

import time
from hydra.utils import instantiate
from omegaconf import OmegaConf


def get_mgrid(sidelen, dim=2):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int"""
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


from spellbook import count_parameters


def get_cameraman_tensor(sidelength):
    pil_img = Image.fromarray(skimage.data.camera())
    transform = Compose([Resize(sidelength), ToTensor(), Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))])
    img = transform(pil_img)
    return img, pil_img


class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img, pil_img = get_cameraman_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)
        self.pil_img = pil_img

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError

        return self.coords, self.pixels


import os
import random
import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def mse_and_psnr(im_a, im_b, data_range=1):
    data_range = 1.0
    mse = F.mse_loss(im_a, im_b)
    psnr = 10 * torch.log10((data_range**2) / mse)
    return mse, psnr


from pathlib import Path

Path.ls = lambda x: list(x.iterdir())


def get_uvs(H, W, device="cpu"):
    xs = torch.linspace(-1, 1, W).to(device)
    ys = torch.linspace(-1, 1, H).to(device)

    x, y = torch.meshgrid(xs, ys, indexing="xy")
    uvs = torch.stack([x, y], dim=-1)
    return uvs


def load_data(cfg):
    if cfg.image == "cameraman":
        cameraman = ImageFitting(256)
        dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)
        model_input, ground_truth = next(iter(dataloader))
        W, H = 256, 256

    else:
        image_path = Path(cfg.image_dir) / cfg.image
        assert image_path.exists()

        image_pil = Image.open(image_path)
        W, H = image_pil.size

        desired_W = cfg.get("image_W")
        if desired_W:
            aspect = H / W
            W = desired_W

            H = int(W * aspect)
            image_pil = image_pil.resize((W, H))

        image = torch.tensor(np.asarray(image_pil)).permute(2, 0, 1) / 255

        # to verify:
        # F.grid_sample(image[None], uvs[None]).rgb
        uvs = get_uvs(H, W, device="cpu")
        model_input, ground_truth = uvs.reshape(1, -1, 2), image.reshape(1, -1, 3)

    return model_input, ground_truth, H, W


def to01(t):
    t = t - t.min()
    t = t / t.max()
    return t


def tensor2pil(t):
    return Image.fromarray((t.detach().permute(1, 2, 0).cpu().clip(0, 1).numpy() * 255).astype(np.uint8))


def pil2tensor(pil):
    return torch.tensor(np.asarray(pil).astype(np.float32) / 255).permute(2, 0, 1)


def imagify_tensor(t, H, W):
    return tensor2pil(to01(t.reshape(-1, H, W).expand(3, H, W)))


def _train_seed(cfg, random_seed=0):
    seed_all(random_seed)
    print("Setting seed to", random_seed)
    is_debug = cfg.get("is_debug")

    project = str(cfg.logging.logger.project).replace(".jpg", "_jpg").replace(".png", "_png")

    if is_debug:
        project = "DEBUG__" + project

    logger = instantiate(
        cfg.logging.logger,
        project=project,
        group=cfg.logging.experiment_name,
        name=f"rs{random_seed}",
    )

    print("*" * 80)
    print("\n")
    print(OmegaConf.to_yaml(cfg))
    print()
    print("*" * 80)

    device = cfg["device"]

    model_input, ground_truth, H, W = load_data(cfg)
    model_input, ground_truth = model_input.to(device), ground_truth.to(device)

    out_features = ground_truth.shape[-1]
    model = instantiate(cfg["model"], out_features=out_features)
    model.to(device)

    total_steps = cfg["total_steps"]
    steps_til_summary = cfg.logging["steps_till_summary"]

    total_params = count_parameters(model)

    logger.log_dict({"total_params": total_params})

    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    for step in range(total_steps):
        model_output = model(model_input)
        mse, psnr = mse_and_psnr(model_output, ground_truth)
        loss = mse

        log_dic = {"step": step, "mse": mse.item(), "psnr": psnr.item()}
        logger.log_dict(log_dic)

        if not step % steps_til_summary:
            print(f"Step {step}, Total loss {loss:0.6f}")
            # img_grad_tensor = gradient(model_output, coords)
            # img_laplacian_tensor = laplace(model_output, coords)

            img = imagify_tensor(model_output, H, W)
            # img_grad = imagify_tensor(img_grad_tensor.norm(dim=-1))
            # img_laplacian = imagify_tensor(img_laplacian_tensor)

            colage = img
            plt.imshow(colage)
            plt.show()

            logger.log_image(colage, name="pred_image")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return model


def train(cfg):
    for random_seed in cfg.random_seed:
        _train_seed(cfg, random_seed=random_seed)
