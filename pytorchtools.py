import torch
import numpy as np
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self,prePath,optimizer,patience=12,verbose=False,delta=0,):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.prePath = prePath
        self.optimizer = optimizer
        self.epoch = 0

    def __call__(self, val_loss, model, epoch):

        score = -val_loss
        self.epoch = epoch

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': model,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': val_loss
            },self.prePath+'checkpoint.pth') # This will store the parameters of the best model obtained so far.
        
        # torch.save(model, 'finish_model.pkl') # This will store the parameters of the best model obtained so far.
        self.val_loss_min = val_loss
