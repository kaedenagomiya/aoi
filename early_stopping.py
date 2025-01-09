import numpy as np
import torch

"""
e.g. usage

```
max_epochs_without_improvement=3
path_save_model='./crepe'
path_save_pth=f'{path_save_model}/crepe_{model_capacity}_best.pth'
# setting params for train
best_val_loss = float('inf')
epochs_without_improvement = 0
accumulation_steps = 4

early_stopping = EarlyStopping(
    patience=max_epochs_without_improvement,
    verbose=True,
    delta=0,
    path=path_save_pth
)

# train
for epoch in range(1, num_epoch + 1):
    for _ in range(num_batches_per_epoch):
    model.train()


    print('[seq] run val')
    model.eval()
    val_loss = 0.0
    running_val_loss = 0.0
    with torch.no_grad():
        for i, (audio, labels) in enumerate(tqdm(val_loader)):
            tmp_val_loss = compute_loss()
            running_val_loss += tmp_val_loss.item() * audio.size(0)
        
    val_loss = running_val_loss / len(val_loader.dataset)
    
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break
```
"""


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience:int=10, verbose:bool=False, delta:float=0, path:str='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        # Check if validation loss is nan
        if np.isnan(val_loss):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            return

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_val_loss - self.delta:
            # Significant improvement detected
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss