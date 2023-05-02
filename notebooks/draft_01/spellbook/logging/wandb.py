import wandb

psnr_metrics_summary = (
    ("psnr", "max"),
    ("mse", "min"),
)


class WandbLogger:
    def __init__(
        self,
        metrics_summary=psnr_metrics_summary,
        **init_kwargs
    ):
        wandb.init(**init_kwargs)

        for metric, summary in metrics_summary:
            wandb.define_metric(metric, summary=summary)

    def log_dict(self, dict_to_log):
        wandb.log(dict_to_log)

    def log_image(self, image, name="img", caption=None):
        wandb.log({name: [wandb.Image(image, caption=caption)]})

class ConsoleLogger:
    def __init__(self, metrics_summary=psnr_metrics_summary):
        self.metrics_summary = metrics_summary
    
    def log_dict(self, dict_to_log):
        print(dict_to_log)

    def log_image(self, image, name, caption=None):
        pass