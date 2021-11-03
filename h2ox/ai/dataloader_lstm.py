import os
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from random import shuffle
from dateutil.relativedelta import relativedelta

from torch.utils.data import DataLoader, Dataset

class w2w_lstm_loader(Dataset):
    """
    Some Docstrings
    
    """
    
    
    def __init__(self, csv_data_path, horizon_days, lead_days, segment, site, perturb=0, shuffle_records=True, targets=True, start_date=None, end_date=None):
        
        self.segment = segment
        self.lead_days = lead_days
        self.horizon_days = horizon_days
        self.targets = targets
        self.site = site
        self.perturb=perturb
        
        ## perturb pos -> temp up, tp down
        ## perturb down -> temp down, tp up
        
        ### load the data
        df = pd.read_csv(csv_data_path).rename(columns={'Unnamed: 0':'date'})
        df = df.groupby('date').nth(0)
        df.index = pd.to_datetime(df.index)
        
        # add 3 months of data for deployment
        max_date = df.index.max()
        df = df.append(
            pd.DataFrame(
                index = pd.date_range(max_date+relativedelta(days=1),max_date+relativedelta(days=92),freq='1d'),
                columns=df.columns,
                data = np.nan
            )
        )
        
        
        ### construct the features
        df['Y'] = df['volume_bcm']
        
                    
        ### add day-of-year cosine
        df['sin_dayofyear']=np.sin((df.index.dayofyear - 1)/365*2*np.pi)
        
        
        ### attach to instance
        self.df = df
        
        
        ### set up the records -> a dict with int_index:shuffled(dt_index)
        
        """valid_idxs = (
            (~pd.isna(df['volume_bcm'])) &
            (~(pd.isna(df[[f'tp_{ii}' for ii in range(15)]]).any(axis=1))) &
            (~pd.isna(df[[f't2m_{ii}' for ii in range(15)]]).any(axis=1)) &
            (~pd.isna(df[[f'Y_{ii}' for ii in range(5, self.targets_forecast,5)]]).any(axis=1))
        )"""
        
        valid_idxs = df.any(axis=1)
        
        #valid_idxs = valid_idxs & (((self.df.index<=idx_dt) & (self.df.index>(idx_dt - relativedelta(days=self.lead_days)))).sum()==)
        
        if segment is not None:
            valid_idxs = valid_idxs & (df['set'].isin(segment))
            
        if start_date is not None:
            valid_idxs = valid_idxs & (df.index >= start_date) & (df.index<end_date)
            
        
        print (self.segment, 'valid records:',valid_idxs.sum())
        self.records = df.loc[valid_idxs].index.tolist()

        if shuffle_records:
            shuffle(self.records)
        self.records = dict(zip(range(len(self.records)),self.records))
        
    def __len__(self):
        return len(self.records.keys())
    
    def __getitem__(self, index):
        
        idx_dt = self.records[index]

        
        ### need three chunks: historic, future_1, future_2
        # historic_x: Y, tp_0, t2m_0, sin_dayofyear for lag_window
        # future_1_x: tp_x, t2m_x, sin_dayofyear for horizon
        # future_2_x: sin_dayofyear for horizon-14
        # Y_future_1: Y
        # Y_future_2: Y
        
        historic_cols = ['Y','tp_0','t2m_0','sin_dayofyear']
        
        X_historic = self.df.loc[((self.df.index<=idx_dt) & (self.df.index>(idx_dt - relativedelta(days=self.lead_days)))),historic_cols].values
        

        arrs = [
            self.df.loc[idx_dt,[f'tp_{ii}' for ii in range(1,15)]].values * (1. + np.arange(14)/13.*self.perturb),
            self.df.loc[idx_dt,[f't2m_{ii}' for ii in range(1,15)]].values ,
            self.df.loc[((self.df.index>idx_dt) & (self.df.index<(idx_dt + relativedelta(days=15)))),'sin_dayofyear'].values
        ]

        X_future_1 = np.stack(arrs).T

        X_future_2 = self.df.loc[((self.df.index>=(idx_dt+relativedelta(days=15))) & (self.df.index<=(idx_dt + relativedelta(days=self.horizon_days)))),'sin_dayofyear'].values.reshape(-1,1)
        
        X = {
            'historic':torch.from_numpy(X_historic.astype(np.float32)),
            'future_1':torch.from_numpy(X_future_1.astype(np.float32)),
            'future_2':torch.from_numpy(X_future_2.astype(np.float32)),
        }
        
        if self.targets:
        
            Y_future_1 = self.df.loc[((self.df.index>idx_dt) & (self.df.index<(idx_dt + relativedelta(days=15)))),'Y'].values
            Y_future_2 = self.df.loc[((self.df.index>=(idx_dt+relativedelta(days=15))) & (self.df.index<=(idx_dt + relativedelta(days=self.horizon_days)))),'Y'].values


            Y = {
                'future_1':torch.from_numpy(Y_future_1.astype(np.float32)),
                'future_2':torch.from_numpy(Y_future_2.astype(np.float32)),
            }

            return X,Y
        else:
            return X
        
        
if __name__=="__main__":
    ### do some tests
    
    root = os.getcwd()
    
    """dataset = w2w_lstm_loader(
        csv_data_path=os.path.join(root,'wave2web_data','kabini_zscore.csv'), 
        horizon_days=90, 
        lead_days=60, 
        segment=['trn'],
    )
    
    
    for ii in np.random.choice(dataset.__len__(),3):
        X, Y = dataset.__getitem__(ii)
        for kk in X.keys():
            print (kk, X[kk].shape)
            
    #for ii in range(dataset.__len__()):
    #    X, Y = dataset.__getitem__(ii)
    #    #print ([(kk,vv.shape) for kk,vv in X.items()], [(kk,vv.shape) for kk,vv in Y.items()])
            
            
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    X,Y = next(iter(loader))
    
    for kk,vv in X.items():
        print (kk,vv.shape)
        
    for kk,vv in Y.items():
        print (kk,vv.shape)

        
    #print ('testing shapes...')
    #for ii in range(loader.__len__()):
    #    X, Y = loader.__getitem__(ii)"""
    
    whole_dataset = w2w_lstm_loader(
        csv_data_path=os.path.join(root, 'wave2web_data',f'kabini_3zscore.csv'), 
        horizon_days=90, 
        lead_days=60, 
        segment=['trn','val','test','deploy'],
        targets=False,
        shuffle_records=False,
    )
    
    loader = DataLoader(whole_dataset, batch_size=40, num_workers=6,shuffle=False)
    for X in loader:
        print ([(kk,xx.shape) for kk,xx in X.items()])
