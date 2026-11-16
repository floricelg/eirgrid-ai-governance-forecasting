"""Chronological, direct-horizon demand forecasting with no future covariates."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data import audit,make_dataset,HORIZONS,OUT,FEATURES

def lstm_fit_predict(Xtr,ytr,Xva,yva,Xte,seed,epochs):
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError('LSTM requires torch; install requirements.txt') from exc
    torch.manual_seed(seed);torch.set_num_threads(2)
    # Feature-history sequences are ordered lagged values, not future observations.
    scaler=StandardScaler().fit(Xtr)
    a=scaler.transform(Xtr).astype('float32')
    b=scaler.transform(Xva).astype('float32')
    c=scaler.transform(Xte).astype('float32')
    mu=float(ytr.mean()); sd=float(ytr.std())
    if sd==0:raise ValueError('Constant training target')
    yy=((ytr-mu)/sd).astype('float32')
    yv=((yva-mu)/sd).astype('float32')
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn=nn.LSTM(1,24,batch_first=True)
            self.head=nn.Linear(24,1)
        def forward(self,x):
            _,(h,_)=self.rnn(x.unsqueeze(-1))
            return self.head(h[-1]).squeeze(-1)
    model=Net();opt=torch.optim.Adam(model.parameters(),lr=.005)
    best=float('inf');wait=0;state=None
    rng=np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        for batch in np.array_split(rng.permutation(len(a)),max(1,int(np.ceil(len(a)/64)))):
            pred=model(torch.tensor(a[batch]))
            loss=((pred-torch.tensor(yy[batch]))**2).mean()
            opt.zero_grad();loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            loss=float(((model(torch.tensor(b))-torch.tensor(yv))**2).mean())
        if loss<best-1e-7:
            best=loss;wait=0
            state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:wait+=1
        if wait>=8:break
    model.load_state_dict(state);model.eval()
    with torch.no_grad():return model(torch.tensor(c)).numpy()*sd+mu

def run(stage='all',seed=13,epochs=60):
    df=audit()
    if stage=='audit':return
    OUT.mkdir(exist_ok=True)
    for label,h in HORIZONS.items():
        X,y,base,origins,times=make_dataset(df,h)
        n=len(y)
        # Chronological 60/20/20 split by target date, with embargo of h between splits.
        train_end=int(n*.6);val_end=int(n*.8)
        train=np.arange(0,train_end)
        validation=np.flatnonzero((origins>=origins[train_end-1]+h)&(np.arange(n)<val_end))
        test=np.flatnonzero(origins>=origins[val_end-1]+h)
        if min(len(train),len(validation),len(test))<5:
            raise ValueError(f'{label}: insufficient independent origins for train/validation/test: '
                             f'{len(train)}/{len(validation)}/{len(test)}. '
                             'For long horizons, use more years or revise the design.')
        predictions={'Seasonal Naive':base[test]}
        models={
            'Linear Regression':make_pipeline(StandardScaler(),Ridge(alpha=10.)),
            'Random Forest':RandomForestRegressor(n_estimators=120,min_samples_leaf=3,
                                                 random_state=seed,n_jobs=2),
            'Gradient Boosting':HistGradientBoostingRegressor(max_iter=120,l2_regularization=1.,
                                                               random_state=seed)}
        for name,model in models.items():
            model.fit(X[train],y[train])
            predictions[name]=model.predict(X[test])
        if stage in ('all','main'):
            predictions['LSTM']=lstm_fit_predict(X[train],y[train],X[validation],y[validation],
                                                  X[test],seed,epochs)
        rows=[]
        for name,pred in predictions.items():
            for i,idx in enumerate(test):
                rows.append({'horizon':label,'lead_steps':h,'model':name,
                             'origin_utc':str(df.utc.iloc[origins[idx]]),
                             'target_utc':str(pd.Timestamp(times[idx])),
                             'actual':float(y[idx]),'prediction':float(pred[i]),
                             'baseline':float(base[idx]),'seed':seed,'partition':'test'})
        output=OUT/f'predictions_{label}.csv'
        pd.DataFrame(rows).to_csv(output,index=False)
        print(f'{label}: train={len(train)}, validation={len(validation)}, test={len(test)}; '
              f'models={len(predictions)}; saved {output}')
    (OUT/'experiment_config.json').write_text(json.dumps({
        'design':'Direct single-target forecasting at 24h, 7d, 14d leads',
        'note':'Not a full 96/672/1344-step forecast path',
        'origin_stride':'96 (daily) for all horizons','split':'chronological 60/20/20 with horizon embargo',
        'baseline':'demand at origin minus 7 days',
        'seed':seed,'epochs_max':epochs,'features':FEATURES,
        'horizons':HORIZONS},indent=2))
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['audit','main','all'],default='all')
    p.add_argument('--seed',type=int,default=13)
    p.add_argument('--epochs',type=int,default=60)
    a=p.parse_args()
    run(a.stage,a.seed,a.epochs)
if __name__=='__main__':main()
