import argparse

from train_pipeline import *
import wandb

def main(sweep_id):
    from train_pipeline import train

    import os
    import torch
    from hydra import initialize, initialize_config_module, initialize_config_dir, compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf


    def load_cfg(overrides=()):
        # with initialize_config_dir(config_dir="/app/notebooks/draft_02/conf"):
        with initialize(version_base=None, config_path="./conf"):
            cfg = compose(config_name='config', overrides=list(overrides))
            return cfg

    from IPython.display import clear_output

    import lovely_tensors as lt
    lt.monkey_patch()

    cfg = load_cfg(overrides=[
        "+exp=06_siren_hp",
        # "model.first_layer_init_c={flic}",
        # "model.init_c={init_c}",
        "+device=cuda:1",
    ])

    # print(OmegaConf.to_yaml(cfg))
    # model, psnr = _train_seed(cfg, cfg.random_seed[0])

    project_name = 'tune_siren__cameraman'


    def _custom_train_seed(cfg, random_seed=0):
        seed_all(random_seed)
        print("Setting seed to", random_seed)

        logger = instantiate(
            cfg.logging.logger,
            project='эє',
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
        batch_size = cfg.get('batch_size', None)

        best_psnr = 0
        optimizer = instantiate(cfg.optimizer, params=model.parameters())

        for step in range(total_steps):
            if batch_size:
                idxs = torch.randint(0, model_input.shape[1], (batch_size,))
                model_input_batch = model_input[:, idxs]
                ground_truth_batch = ground_truth[:, idxs]
            else:
                model_input_batch = model_input
                ground_truth_batch = ground_truth

            model_output_batch = model(model_input_batch)
            mse, psnr = mse_and_psnr(model_output_batch, ground_truth_batch)
            loss = mse

            psnr = psnr
            
            if best_psnr < psnr:
                best_psnr = psnr
            # log_dic = {"step": step, "mse": mse.item(), "psnr": psnr.item()}
            # logger.log_dict(log_dic)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        return model, best_psnr


    def _train_for_hparams():
        wandb.init(project=project_name)
        wandb_config = wandb.config
        cfg = load_cfg(overrides=[
            "+exp=06_siren_hp",
            # "total_steps=12",
            f"model.first_layer_init_c={wandb_config.flic}",
            f"model.init_c={wandb_config.initc}",
            "+device=cuda:1",
        ])
        
        print(cfg.logging.experiment_name)

        model, last_psnr = _custom_train_seed(cfg, cfg.random_seed[0])
        print('last_psnr', last_psnr)
        wandb.log({'last_psnr': last_psnr})
        
    wandb.agent(sweep_id, function=_train_for_hparams, count=100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python CLI App")
    parser.add_argument("--sweep_id", help="ID of the sweep", required=True)
    args = parser.parse_args()
    
    wandb.login()
    main(args.sweep_id)
