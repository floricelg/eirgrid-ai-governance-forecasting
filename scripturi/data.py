"""Read EirGrid Excel and construct strictly causal direct-horizon features."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'date'
OUT=ROOT/'rezultate'
HORIZONS={'24h':96,'7d':672,'14d':1344}
LAGS=[1,4,96,192,672]
FEATURES=[f'lag_{n}' for n in LAGS]+['mean_96','mean_672','std_96',
          'hour_sin','hour_cos','week_sin','week_cos','year_sin','year_cos']
def source():
    paths=sorted(DATA.glob('*.xlsx'))
    if len(paths)!=1:
        raise ValueError('Place exactly one EirGrid .xlsx in date/. Found '+str(len(paths)))
    return paths[0]
def load():
    path=source()
    df=pd.read_excel(path,sheet_name='System Data',usecols=['DateTime','GMT Offset','IE Demand'])
    df['DateTime']=pd.to_datetime(df['DateTime'],errors='coerce')
    df['IE Demand']=pd.to_numeric(df['IE Demand'],errors='coerce')
    if df[['DateTime','IE Demand']].isna().any().any():
        raise ValueError('Missing or invalid timestamps/demand; inspect raw data before modelling')
    if (df['IE Demand']<=0).any():
        raise ValueError('Nonpositive demand detected: MAPE would be undefined')
    # EirGrid supplies local timestamps and GMT offsets; UTC removes DST repeated-hour ambiguity.
    offset=pd.to_numeric(df['GMT Offset'],errors='coerce')
    if offset.isna().any(): raise ValueError('Invalid GMT Offset')
    if not set(offset.unique()).issubset({0,1,-1,0.0,1.0,-1.0}):
        raise ValueError('Unexpected GMT Offset units; inspect source before proceeding')
    utc=df['DateTime']-pd.to_timedelta(offset,unit='h')
    ordered=pd.DataFrame({'utc':utc,'demand':df['IE Demand']}).sort_values('utc')
    if ordered.utc.duplicated().any():raise ValueError('Duplicate UTC timestamps')
    if not ordered.utc.diff().dropna().eq(pd.Timedelta(minutes=15)).all():
        raise ValueError('UTC series has missing or irregular quarter-hours; no silent imputation')
    return ordered.reset_index(drop=True),path
def feature_at(values,t,ts):
    # Only observations at or before origin t may enter the feature vector.
    x=[values[t-k] for k in LAGS]
    x += [values[t-95:t+1].mean(),values[t-671:t+1].mean(),values[t-95:t+1].std()]
    d=ts
    hour=d.hour+d.minute/60
    week=d.dayofweek+hour/24
    year=d.dayofyear+hour/24
    x += [np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),
          np.sin(2*np.pi*week/7),np.cos(2*np.pi*week/7),
          np.sin(2*np.pi*year/365.25),np.cos(2*np.pi*year/365.25)]
    return x
def make_dataset(df,horizon):
    y=df.demand.to_numpy(dtype=float)
    ts=df.utc.dt
    times=df.utc.to_numpy()
    # Daily origins for all leads; targets may overlap across long horizons.
    # Dependence must be considered in inferential tests.
    stride=96
    origins=np.arange(672,len(y)-horizon,stride)
    X=np.array([feature_at(y,t,df.utc.iloc[t]) for t in origins],dtype=np.float64)
    target=y[origins+horizon]
    # A baseline observed by the origin: same quarter-hour of a previous week.
    # For h=14d, last available same weekly slot is y[t-672].
    baseline=y[origins-672]
    return X,target,baseline,origins,times[origins+horizon]
def audit():
    OUT.mkdir(exist_ok=True)
    df,path=load()
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    report={'file':path.name,'sha256':sha,'rows':len(df),'start_utc':str(df.utc.min()),
            'end_utc':str(df.utc.max()),'missing_demand':int(df.demand.isna().sum()),
            'quarter_hourly_utc':True,'minimum_demand':float(df.demand.min()),
            'maximum_demand':float(df.demand.max()),'mean_demand':float(df.demand.mean())}
    (OUT/'data_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return df
