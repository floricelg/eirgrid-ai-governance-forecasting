"""Independent checks of forecast lineage, causality, and metric consistency."""
import numpy as np
import pandas as pd
from data import load,make_dataset,HORIZONS,OUT
def main():
    df,_=load()
    for label,h in HORIZONS.items():
        X,y,baseline,origins,times=make_dataset(df,h)
        assert np.all(np.isfinite(X)) and np.all(np.isfinite(y))
        assert np.all(origins+h<len(df))
        assert np.allclose(baseline,df.demand.to_numpy()[origins-672])
        file=OUT/f'predictions_{label}.csv'
        pred=pd.read_csv(file)
        assert pred.model.nunique()>=4
        assert pred.groupby('model').size().nunique()==1
        assert pred.groupby('model').target_utc.nunique().nunique()==1
        assert np.isfinite(pred[['actual','prediction','baseline']]).all().all()
        for model,group in pred.groupby('model'):
            assert not group.target_utc.duplicated().any(),model
            assert (pd.to_datetime(group.origin_utc)<pd.to_datetime(group.target_utc)).all()
            assert (pd.to_datetime(group.target_utc)-pd.to_datetime(group.origin_utc)
                    ==pd.Timedelta(minutes=15*h)).all()
            assert np.allclose(group.baseline.to_numpy(),group.baseline.to_numpy())
        print('PASS',label,len(pred),'forecast records')
    print('Lineage and structural checks passed. They do not certify scientific validity.')
if __name__=='__main__':main()
