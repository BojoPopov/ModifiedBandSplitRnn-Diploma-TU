from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import StochasticWeightAveraging
from datasetloader import MusicDataModule
from ema_lightning_bandsplit_no_learnableSTFT_no_NormOnBand import BandSplitRNN
import torch

torch.set_float32_matmul_precision('medium')  # Or 'high'
# from pytorch_lightning.callbacks import Callback
#
# class DynamicLRScheduler(Callback):
#     def __init__(self, initial_lr=1e-6, factor_up=10, factor_down=0.1):
#         """
#         Custom LR scheduler with warmup logic.
#
#         Args:
#             initial_lr (float): Starting learning rate.
#             factor_up (float): Factor to multiply LR during warmup when loss improves.
#             factor_down (float): Factor to divide LR when loss stagnates post-warmup.
#         """
#         self.initial_lr = initial_lr
#         self.factor_up = factor_up
#         self.factor_down = factor_down
#         self.prev_loss = None  # Store the previous epoch's loss
#         self.warmup = True  # Warmup phase flag
#
#     def on_train_start(self, trainer, pl_module):
#         """Initialize the learning rate at the start of training."""
#         optimizer = trainer.optimizers[0]
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = self.initial_lr
#         print(f"Starting training with LR = {self.initial_lr:.8f}, Warmup = {self.warmup}")
#
#     def on_train_epoch_end(self, trainer, pl_module):
#         """Adjust the learning rate at the end of each epoch based on avg loss."""
#         avg_loss = trainer.callback_metrics.get('train_loss').item()
#         optimizer = trainer.optimizers[0]
#
#         if self.prev_loss is None:
#             # First epoch, initialize previous loss
#             self.prev_loss = avg_loss
#             print(f"First epoch completed. Setting prev_loss = {avg_loss:.5f}")
#             return
#
#         if self.warmup:
#             # Warmup phase: increase LR if loss improves
#             if avg_loss <= self.prev_loss:
#                 new_lr = self._update_lr(optimizer, self.factor_up)
#                 print(f"[Warmup] Loss improved to {avg_loss:.5f}, increasing LR to {new_lr:.8f}")
#             else:
#                 # Loss did not improve: End warmup
#                 self.warmup = False
#                 print(f"[Warmup] Loss stagnated/worsened at {avg_loss:.5f}, ending warmup. LR = {self._get_current_lr(optimizer):.8f}")
#         else:
#             # Post-warmup phase: reduce LR if loss did not improve
#             if avg_loss > self.prev_loss:
#                 new_lr = self._update_lr(optimizer, self.factor_down)
#                 print(f"[Post-Warmup] Loss stagnated/worsened at {avg_loss:.5f}, reducing LR to {new_lr:.8f}")
#             else:
#                 # Loss improved: keep LR the same
#                 print(f"[Post-Warmup] Loss improved to {avg_loss:.5f}, keeping LR at {self._get_current_lr(optimizer):.8f}")
#
#         # Update previous loss for the next epoch
#         self.prev_loss = avg_loss
#
#     def _update_lr(self, optimizer, factor):
#         """Update the learning rate by a factor."""
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = param_group['lr'] * factor
#         return param_group['lr']
#
#     def _get_current_lr(self, optimizer):
#         """Retrieve the current learning rate."""
#         return optimizer.param_groups[0]['lr']
#
# class SWACallback(Callback):
#     """
#     Callback for managing Stochastic Weight Averaging (SWA).
#     """
#     def __init__(self):
#         self.swa_helper = None
#
#     def on_fit_start(self, trainer, pl_module):
#         self.swa_helper = SWA(pl_module)
#
#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
#         # Update SWA model after each batch
#         self.swa_helper.update(pl_module)
#
#     def on_validation_start(self, trainer, pl_module):
#         # Apply SWA weights before validation
#         self.swa_helper.apply_swa(pl_module)
#
#     def on_validation_end(self, trainer, pl_module):
#         # Restore original weights after validation (if applicable)
#         self.swa_helper.restore(pl_module)
#
#     def save_swa_model(self, filepath: str):
#         """Saves the SWA model weights to the specified filepath."""
#         swa_weights = {
#             name: param.clone().detach().cpu()
#             for name, param in self.swa_helper.get_swa_weights().items()
#         }
#         torch.save(swa_weights, filepath)
#
# class EMACallback(Callback):
#     def __init__(self, decay=0.999):
#         self.decay = decay
#         self.ema_helper = None
#
#     def on_fit_start(self, trainer, pl_module):
#         self.ema_helper = EMA(pl_module, self.decay)
#
#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
#         self.ema_helper.update(pl_module)
#
#     def on_validation_start(self, trainer, pl_module):
#         self.ema_helper.apply_shadow(pl_module)
#
#     def on_validation_end(self, trainer, pl_module):
#         self.ema_helper.restore(pl_module)
#
#     def save_ema_model(self, filepath: str):
#         """Saves the EMA model weights to the specified filepath."""
#         ema_weights = {
#             name: param.clone().detach().cpu()
#             for name, param in self.ema_helper.shadow.items()
#         }
#         torch.save(ema_weights, filepath)

# Define Band Splits
splits_v7 = [
    (1000, 100),
    (4000, 250),
    (8000, 500),
    (16000, 1000),
    (20000, 2000),
]

# Initialize Dataset and Model
root_dir = "D:/User/Desktop/musdb_normal/eight_seconds"
data_module = MusicDataModule(root_dir, batch_size=4)

# Initialize model with SWA and EMA enabled
model = BandSplitRNN(
    bandsplits=splits_v7,
    num_layers=5,
    lr=1e-4
).to('cuda')
# dynamic_lr_callback = DynamicLRScheduler(initial_lr=1e-6, factor_up=10, factor_down=0.1)

if __name__ == "__main__":
    # Define trainer with EMA callback
    # ema_callback = EMACallback(decay=0.999)
    # swa_callback = StochasticWeightAveraging(swa_lrs=5e-4)
    trainer = Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        precision='16-mixed',
        gradient_clip_val=1,
        gradient_clip_algorithm='norm',
        enable_checkpointing=True,
        # callbacks=[ema_callback]
    )

    # Fit the model
    trainer.fit(model, datamodule=data_module)

    # Save EMA model
    # ema_callback.save_ema_model("ema_model_weights.pth")
    #
    # # Save original model
    # torch.save(model.state_dict(), "original_model_weights.pth")

    # swa_callback.save_swa_model('swa_model.pth')