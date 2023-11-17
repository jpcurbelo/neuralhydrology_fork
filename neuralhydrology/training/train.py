from neuralhydrology.training.basetrainer_gsus import BaseTrainer
from neuralhydrology.utils.config import Config


def start_training(cfg: Config):
    """Start model training.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.

    """

    # MC-LSTM is a special case, where the head returns an empty string but the model is trained as regression model.
    if cfg.head.lower() in ['regression', 'gmm', 'umal', 'cmal', '']:
        trainer = BaseTrainer(cfg=cfg)
    else:
        raise ValueError(f"Unknown head {cfg.head}.")
    
    print("\n trainer.initialize_training() \n")
    trainer.initialize_training()
    print("\n trainer.train_and_validate() \n")
    trainer.train_and_validate()
