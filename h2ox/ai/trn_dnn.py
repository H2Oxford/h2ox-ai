import logging, os
logging.basicConfig(level=logging.INFO)
import pandas as pd
from datetime import timedelta, datetime
from torchsummary import summary

from tqdm import tqdm

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from src.w2w.dataloader_dnn import w2w_dnn_loader
from matplotlib.collections import LineCollection

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)


class W2W_DNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # activation?
        return x
    
    
def whole_visualise(model, whole_loader, device, savepath):
    
    # collate results
    model.eval()
    Y_results = []
    Y_hat_results = []
    with torch.no_grad():
        val_loss = 0
        for batch_idx, (X, Y) in enumerate(whole_loader):

            X, Y = X.to(device), Y.to(device)

            Y_hat = model(X)
            
            Y_results.append(Y.detach().to('cpu').numpy())
            Y_hat_results.append(Y_hat.detach().to('cpu').numpy())
            
    
    results_df = pd.DataFrame(
        index =list(whole_loader.dataset.records.values()), 
        data=np.concatenate(Y_hat_results)*whole_loader.dataset.Y_std + whole_loader.dataset.Y_mean,
        columns = [timedelta(days=ii) for ii in [1]+list(range(5,whole_loader.dataset.targets_forecast,5))]
    )
    
    results_df = results_df.sort_index()    
    results_df.to_csv(os.path.splitext(savepath)[0]+'.csv')
    
    #len(range(5,whole_loader.dataset.targets_forecast,5))
    fig, axs = plt.subplots(4,1,figsize=(18,12))
        
    for ii_a, forecast_offset in enumerate([5,30,60,90]):
        
        
        axs[ii_a].plot(
            results_df.index.values.astype(np.int64) // 10 ** 9,
            whole_loader.dataset.df.loc[results_df.index,'Y']*whole_loader.dataset.Y_std + whole_loader.dataset.Y_mean,
            c='k',
            linestyle='solid',
            lw=2,
            label='observed'
        )
        
        axs[ii_a].plot(
            (results_df.index.values + np.timedelta64(forecast_offset,'D')).astype(np.int64) // 10 ** 9,
            results_df.loc[:,timedelta(days=forecast_offset)],
            c='r',
            linestyle='solid',
            lw=2,
            label='projection',
        )
        
        axs[ii_a].set_ylabel(f'Projection: {forecast_offset} Days')
        
    fig.suptitle(savepath)
    
    plt.savefig(savepath)
    
def eval_r2(model, test_loader, device):
    
    # collate results
    model.eval()
    Y_results = []
    Y_hat_results = []
    with torch.no_grad():
        val_loss = 0
        for batch_idx, (X, Y) in enumerate(test_loader):

            X, Y = X.to(device), Y.to(device)

            Y_hat = model(X)
            
            Y_results.append(Y.detach().to('cpu').numpy())
            Y_hat_results.append(Y_hat.detach().to('cpu').numpy())
            
    Y_results = np.concatenate(Y_results)*test_loader.dataset.Y_std + test_loader.dataset.Y_mean
    Y_hat_results = np.concatenate(Y_hat_results)*test_loader.dataset.Y_std + test_loader.dataset.Y_mean
    
    print ('Y_results shape',Y_results.shape)
    
    for jj in range(Y_results.shape[1]):
        print (f'day {jj} r^2 score:', r2_score(Y_results[:,jj], Y_hat_results[:,jj]))
    

def eval_visualise(model, val_loader, device, savepath):
    
    # collate results
    model.eval()
    Y_results = []
    Y_hat_results = []
    with torch.no_grad():
        val_loss = 0
        for batch_idx, (X, Y) in enumerate(val_loader):

            X, Y = X.to(device), Y.to(device)

            Y_hat = model(X)
            
            Y_results.append(Y.detach().to('cpu').numpy())
            Y_hat_results.append(Y.detach().to('cpu').numpy())
            
    
    results_df = pd.DataFrame(
        index =list(val_loader.dataset.records.values()), 
        data=np.concatenate(Y_hat_results)*val_loader.dataset.Y_std + val_loader.dataset.Y_mean,
        columns = [timedelta(days=ii) for ii in [1]+list(range(5,val_loader.dataset.targets_forecast,5))]
    )
    
    results_df = results_df.sort_index()
    
    val_starts = val_loader.dataset.df.loc[(val_loader.dataset.df['segment']=='val') & (val_loader.dataset.df['segment']!=val_loader.dataset.df['segment'].shift(1)),'segment'].index.values
    val_stops = val_loader.dataset.df.loc[(val_loader.dataset.df['segment']=='val') & (val_loader.dataset.df['segment']!=val_loader.dataset.df['segment'].shift(-1)),'segment'].index.values
    val_segments = list(zip(val_starts, val_stops))
    
    lines = []
    for idx, row in results_df.iterrows():
        xs = [(idx+cc).timestamp() for cc in row.index]
        ys = row.values.tolist()
        lines.append(list(zip(xs,ys)))
    
    fig, axs = plt.subplots(6,1,figsize=(18,12))
    
    
    for ii_a in range(axs.shape[0]):
        
        
    
        start_dt = val_segments[ii_a][0]
        end_dt = val_segments[ii_a][1]
        
        lc = LineCollection(lines, colors='lightgray', alpha=0.5, linestyle='solid')
        
        axs[ii_a].add_collection(lc)
    
        axs[ii_a].plot(
            results_df.loc[(results_df.index>=start_dt) & (results_df.index<end_dt )].index.values.astype(np.int64) // 10 ** 9,
            val_loader.dataset.df.loc[results_df.loc[(results_df.index>=start_dt) & (results_df.index<end_dt)].index,'Y']*val_loader.dataset.Y_std + val_loader.dataset.Y_mean,
            c='k',
            linestyle=':',
            lw=2,
        )
        
        axs[ii_a].plot(
            val_loader.dataset.df.loc[(val_loader.dataset.df.index>=end_dt) & (val_loader.dataset.df.index<(end_dt+np.timedelta64(60,'D')))].index.values.astype(np.int64) // 10 ** 9,
            val_loader.dataset.df.loc[(val_loader.dataset.df.index>=end_dt) & (val_loader.dataset.df.index<(end_dt+np.timedelta64(60,'D'))),'Y']*val_loader.dataset.Y_std + val_loader.dataset.Y_mean,
            c='r',
            linestyle=':',
            lw=2,
        )
        
        start_ts  = (start_dt - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        end_ts  = (end_dt+np.timedelta64(60,'D') - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        
        print (start_dt, end_dt,type(start_dt), type(end_dt),start_ts, end_ts)
        axs[ii_a].set_xlim(start_ts, end_ts)
        
    
    plt.savefig(savepath)
    
    
    

def train(model, trn_loader, val_loader, whole_loader, test_loader, optimizer, PARAMS, device, savepath, verbose=True):
    
    n_iter = 0
    for epoch in range(1, PARAMS['EPOCHS'] + 1):
        
        if verbose:
            pbar = tqdm(desc = f'EPOCH:{epoch}', total=len(trn_loader)+len(val_loader))
            
        trn_loss = 0
        
        for batch_idx, (X, Y) in enumerate(trn_loader):
            
            model.train()
            
            X, Y = X.to(device), Y.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            Y_hat = model(X)
            
            
            loss = F.mse_loss(Y_hat, Y)
            #weights = torch.from_numpy(np.stack([np.arange(Y.shape[1],0,-1)/np.arange(Y.shape[1],0,-1).sum()]*X.shape[0])).to(device)
            #loss = weighted_mse_loss(Y_hat, Y, weights)
            
            loss.backward()
            optimizer.step()
        
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                
            trn_loss +=loss.detach().item()
                
            if verbose:
                pbar.update(PARAMS['BATCH_SIZE'])
                pbar.set_description(f'EPOCH:{epoch}, trn_loss:{trn_loss:.3f}')
                    
            del loss, X, Y
        
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch_idx, (X, Y) in enumerate(val_loader):

                X, Y = X.to(device), Y.to(device)
                
                Y_hat = model(X)

                loss = F.mse_loss(Y_hat, Y)
                
                #weights = torch.from_numpy(np.stack([np.arange(Y.shape[1],0,-1)/np.arange(Y.shape[1],0,-1).sum()]*X.shape[0])).to(device)
                #loss = weighted_mse_loss(Y_hat, Y, weights)
                
                val_loss += loss.detach().item()
                
                if verbose:
                    pbar.update(PARAMS['BATCH_SIZE'])
                    pbar.set_description(f'EPOCH:{epoch}, trn_loss:{trn_loss:.3f}, val_loss:{val_loss:.3f}')
                
                del loss, X, Y
                
        if verbose:
            pbar.close()
            
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

            
    #eval_visualise(model, val_loader, device, savepath)
    
    whole_visualise(model, whole_loader, device, savepath)
    
    eval_r2(model, test_loader, device)


    
def main(PARAMS):
    
    root = os.getcwd()
    
    device = torch.device(PARAMS['DEVICE'])
    
    trn_dataset = w2w_dnn_loader(
        csv_data_path=os.path.join(root, 'wave2web_data',f'{PARAMS["SITE"]}_18x2mo_split.csv'), 
        targets_forecast=91, # days 
        lag_windows=[5,15,30,60,100], # days
        segment=None,
        autoregressive=True,
        normalise=True,
        start_date = datetime(year=2014,month=1,day=1),
        end_date = datetime(year=2018,month=12,day=31),
    )
    val_dataset = w2w_dnn_loader(
        csv_data_path=os.path.join(root, 'wave2web_data',f'{PARAMS["SITE"]}_18x2mo_split.csv'), 
        targets_forecast=91, # days 
        lag_windows=[5,15,30,60,100], # days
        segment=None,
        autoregressive=True,
        normalise=True,
        start_date = datetime(year=2011,month=1,day=1),
        end_date = datetime(year=2013,month=12,day=31),
    )
    
    whole_dataset = w2w_dnn_loader(
        csv_data_path=os.path.join(root, 'wave2web_data',f'{PARAMS["SITE"]}_18x2mo_split.csv'), 
        targets_forecast=91, # days 
        lag_windows=[5,15,30,60,100], # days
        segment=None,
        autoregressive=True,
        normalise=True
    )
    
    test_dataset = w2w_dnn_loader(
        csv_data_path=os.path.join(root, 'wave2web_data',f'{PARAMS["SITE"]}_18x2mo_split.csv'), 
        targets_forecast=91, # days 
        lag_windows=[5,15,30,60,100], # days
        segment=None,
        autoregressive=True,
        normalise=True,
        start_date = datetime(year=2019,month=1,day=1),
        end_date = datetime(year=2020,month=12,day=31),
    )
    
    trn_loader = DataLoader(
        trn_dataset, 
        batch_size=PARAMS['BATCH_SIZE'], 
        shuffle=False,
        num_workers=PARAMS['DATALOADER_WORKERS'], 
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=PARAMS['BATCH_SIZE'], 
        shuffle=False,
        num_workers=PARAMS['DATALOADER_WORKERS'], 
    )
    whole_loader = DataLoader(
        whole_dataset, 
        batch_size=PARAMS['BATCH_SIZE'], 
        shuffle=False,
        num_workers=PARAMS['DATALOADER_WORKERS'], 
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=PARAMS['BATCH_SIZE'], 
        shuffle=False,
        num_workers=PARAMS['DATALOADER_WORKERS'], 
    )
    
    _X, _Y = trn_dataset.__getitem__(0)
    
    model = W2W_DNN(
        input_dim  = _X.shape[0], 
        output_dim = _Y.shape[0]
    )
    
    model = model.to(device)
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=PARAMS['LR'])
    
    train(model, trn_loader, val_loader, whole_loader, test_loader, optimizer, PARAMS, device=device, savepath=os.path.join(root,PARAMS['SITE']+'_DNN_break_timeseries.png'), verbose=True)
    
    
if __name__=="__main__":
    
    for kk in ['Kabini','Harangi','Hemavathy','Krisharaja Sagar']:
    
        PARAMS = dict(
            SITE=kk,
            BATCH_SIZE=128,
            DATALOADER_WORKERS = 6,
            LR=0.0005,
            DEVICE='cuda',
            EPOCHS=25,
        )

        main(PARAMS)
