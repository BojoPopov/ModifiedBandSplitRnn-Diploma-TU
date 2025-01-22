import numpy as np
from optuna.samplers import GridSampler
from pytorch_lightning import Trainer
from datasetloader import MusicDataModule
from ema_lightning_bandsplit_no_learnableSTFT_no_NormOnBand import BandSplitRNN
import torchaudio
import optuna
import torch

torch.set_float32_matmul_precision('high')  # Or 'medium'
print(str(torchaudio.list_audio_backends()))



splits_v7 = [
        (1000, 100),
        (4000, 250),
        (8000, 500),
        (16000, 1000),
        (20000, 2000),
    ]
def objective(trial):
    # Suggest hyperparameters
    act1 = trial.suggest_categorical("act1", ["tanh","relu","sigmoid","gelu","selu",'none'])
    act2 = trial.suggest_categorical("act2", ["tanh","relu","sigmoid","gelu","selu",'none'])
    act3 = trial.suggest_categorical("act3", ["tanh","relu","sigmoid","gelu","selu",'none'])
    act4 = trial.suggest_categorical("act4", ["tanh","relu","sigmoid","gelu","selu",'none'])
    act5 = trial.suggest_categorical("act5", ["tanh","relu","sigmoid","gelu","selu",'none'])
    act6 = trial.suggest_categorical("act6", ["tanh","relu","sigmoid","gelu","selu",'none'])
    act7 = trial.suggest_categorical("act7", ["tanh","relu","sigmoid","gelu","selu",'none'])
    act8 = trial.suggest_categorical("act8", ["tanh","relu","sigmoid","gelu","selu",'none'])
    # act9 = trial.suggest_categorical("act9", ["tanh","relu","sigmoid","gelu","softmax","selu",'none'])
    # act10 = trial.suggest_categorical("act10", ["tanh","relu","sigmoid","gelu","softmax","selu",'none'])
    norm1 = trial.suggest_categorical("norm1", ["layernorm","rmsnorm",'none'])
    norm2 = trial.suggest_categorical("norm2", ["layernorm","rmsnorm",'none'])
    norm3 = trial.suggest_categorical("norm3", ["layernorm","rmsnorm",'none'])
    norm4 = trial.suggest_categorical("norm4", ["layernorm","rmsnorm",'none'])
    # norm5 = trial.suggest_categorical("norm5", ["batchnorm2d","instancenorm2d","layernorm","rmsnorm","groupnorm",'none'])
    # act1 = trial.suggest_categorical("act1", ["prelu","leakyrelu","hardswish","mish","silu","celu","elu","selu","rrelu","relu6","hardshrink","hardsigmoid","hardtanh","logsigmoid","logsoftmax","softmin","softplus","softshrink","softsign","tanhshrink"])
    # Instantiate the data module and model
    root_dir = "D:/User/Desktop/musdb_normal/eight_seconds"
    data_module = MusicDataModule(root_dir, batch_size=4)
    model = BandSplitRNN(
        bandsplits=splits_v7,
        act1='elu',
        act2='none',
        act3='elu',
        act4='relu',
        act5='silu',
        act6='none',
        act7='hardtanh',
        act8='celu',
        act9='relu6',
        act10='selu',
        norm1='groupnorm',
        norm2='none',
        norm3='rmsnorm',
        norm4='none',
        norm5='none',
        act_mlp='selu',
        act_glu_mlp='sigmoid',
        act_cbam='sigmoid',
        act_glu_cbam='sigmoid',
        act11=act1,
        act12=act2,
        act13=act3,
        act14=act4,
        act15=act5,
        act16=act6,
        act17=act7,
        act18=act8,
        norm6=norm1,
        norm7=norm2,
        norm8=norm3,
        norm9=norm4,
        fc_dim=128,
        rnn_dim=256,
        mlp_dim=512

        ).to('cuda')

    # Define the trainer and start training
    trainer = Trainer(max_epochs=1 , accelerator="gpu", devices=1, precision='16-mixed', gradient_clip_val=3, gradient_clip_algorithm='norm', enable_checkpointing=False)
    # Training
    try:
        # Training
        trainer.fit(model, datamodule=data_module)

        # Retrieve final training loss
        train_loss = trainer.callback_metrics["train_loss"].item()

        # Check for NaN or Inf in the final loss
        if not np.isfinite(train_loss):
            raise ValueError("NaN or Inf detected in final training loss.")

        return train_loss
    except (ValueError, RuntimeError) as e:
        # Handle numerical issues or training errors
        trial.set_user_attr("error", str(e))
        raise optuna.exceptions.TrialPruned(f"Trial pruned due to numerical issues: {str(e)}")
if __name__ == "__main__":
    # Create an Optuna study and optimize
    # search_space = {"act1":['gelu','hardswish','mish','silu'],"act2":['tanh','softsign','hardtanh']}
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1000)

    # Print the best hyperparameters
    print("Best hyperparameters:", study.best_params)