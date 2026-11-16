"""Generate H1/H2/H3 tables from saved out-of-sample predictions."""
from pathlib import Path
import numpy as np
import pandas as pd
from data import OUT,HORIZONS
TAUS=[.5,.7,.9]
def pinball(y,p,tau):
    e=y-p
    return np.maximum(tau*e,(tau-1)*e)
def dm_hac(d,max_lag):
    """Approximate two-sided normal DM test with Bartlett HAC; report sample size."""
    from scipy.stats import norm
    d=np.asarray(d,dtype=float);n=len(d)
    if n<8:return np.nan
    x=d-d.mean()
    lag=min(max_lag,n//4)
    variance=np.dot(x,x)/n
    for k in range(1,lag+1):
        gamma=np.dot(x[k:],x[:-k])/n
        variance+=2*(1-k/(lag+1))*gamma
    if variance<=0:return np.nan
    return float(2*norm.sf(abs(d.mean()/np.sqrt(variance/n))))
def main():
    OUT.mkdir(exist_ok=True)
    tables=OUT/'tabele';tables.mkdir(exist_ok=True)
    rows=[]; risk=[]; dm=[]
    for horizon,h in HORIZONS.items():
        file=OUT/f'predictions_{horizon}.csv'
        if not file.exists():raise FileNotFoundError(file)
        df=pd.read_csv(file)
        base=df[df.model=='Seasonal Naive'].sort_values('target_utc')
        for name,group in df.groupby('model'):
            group=group.sort_values('target_utc')
            if not np.array_equal(group.target_utc.to_numpy(),base.target_utc.to_numpy()):
                raise ValueError('Forecast target dates differ between models')
            y=group.actual.to_numpy();p=group.prediction.to_numpy()
            err=y-p; be=base.actual.to_numpy()-base.prediction.to_numpy()
            mae=np.mean(np.abs(err));base_mae=np.mean(np.abs(be))
            rows.append(dict(horizon=horizon,model=name,n=len(y),MAE=mae,
                MAPE_percent=100*np.mean(np.abs(err)/y),
                relative_MAE_gain_percent=100*(base_mae-mae)/base_mae))
            for tau in TAUS:
                risk.append(dict(horizon=horizon,model=name,tau=tau,
                                 mean_pinball_loss=float(np.mean(pinball(y,p,tau)))))
            if name!='Seasonal Naive':
                # Loss differential is MAE(model)-MAE(baseline).
                d=np.abs(err)-np.abs(be)
                dm.append(dict(horizon=horizon,model=name,n=len(y),
                               mean_loss_difference=float(d.mean()),
                               DM_p_approx=dm_hac(d,max_lag=min(14,len(y)//4)),
                               note='Exploratory HAC approximation; overlapping daily forecast origins'))
    pd.DataFrame(rows).to_csv(tables/'H1_H2_accuracy.csv',index=False)
    pd.DataFrame(risk).to_csv(tables/'H3_pinball.csv',index=False)
    pd.DataFrame(dm).to_csv(tables/'DM_exploratory.csv',index=False)
    print('Generated H1/H2, H3 and exploratory DM tables.')
    print('H3 scores fixed point predictions; NOT calibrated quantile forecasts.')
if __name__=='__main__':main()
