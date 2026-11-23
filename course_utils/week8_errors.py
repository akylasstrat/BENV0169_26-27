"""Isolated Week 8 error channels around the unchanged Day 1 controller.

The evaluator constructs explicit forecast tables and paired innovations.
The controller receives only those tables, current nowcasts and observations.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .week8_control import run_experiment, realised_metrics


@dataclass(frozen=True)
class ErrorCase:
    forecast_std_c: float = 0.
    forecast_bias_c: float = 0.
    loss_multiplier: float = 1.
    sensor_std_c: float = 0.
    sensor_bias_c: float = 0.
    process_std_c: float = 0.
    drift_after_step: int = 0

    @property
    def stochastic(self):
        return any(x > 0 for x in (self.forecast_std_c,self.sensor_std_c,self.process_std_c))


CASES={
    'C0 No-error reference':ErrorCase(),
    'C1 Forecast noise':ErrorCase(forecast_std_c=1.),
    'C2 Forecast bias':ErrorCase(forecast_bias_c=1.5),
    'C3 Model error':ErrorCase(loss_multiplier=1.3),
    'C4 Sensor noise':ErrorCase(sensor_std_c=.2),
    'C5 Sensor bias':ErrorCase(sensor_bias_c=.4),
    'C6 Process noise':ErrorCase(process_std_c=.05),
    'C7 Combined errors':ErrorCase(forecast_std_c=1.,forecast_bias_c=1.5,
        loss_multiplier=1.3,sensor_std_c=.2,sensor_bias_c=.4,process_std_c=.05),
    'C8 Midday model drift':ErrorCase(loss_multiplier=1.3,drift_after_step=24),
}
CORE_CASES=list(CASES)[:8]


def innovations(seed,M=48,N=48):
    """Separate streams indexed before a controller is run; never draw on observe."""
    process,sensor,forecast=np.random.SeedSequence(int(seed)).spawn(3)
    return {'process':np.random.default_rng(process).normal(size=M),
        'sensor':np.random.default_rng(sensor).normal(size=M+1),
        'forecast':np.random.default_rng(forecast).normal(size=(M,N-1))}


def validate_case(case,model):
    if not isinstance(case,ErrorCase):raise TypeError('Use an explicit ErrorCase.')
    if not all(np.isfinite(list(asdict(case).values()))):raise ValueError('Nonfinite case setting.')
    if min(case.forecast_std_c,case.sensor_std_c,case.process_std_c)<0:raise ValueError('Negative noise scale.')
    if not 0<model['loss_coefficient_per_step']*case.loss_multiplier<1:
        raise ValueError('Controlled-reference loss coefficient must lie in (0,1).')
    if not isinstance(case.drift_after_step,int) or not 0<=case.drift_after_step<48:
        raise ValueError('Drift onset must be a named reporting interval.')


def forecast_table(operation,day,case,draws,M=48,N=48,dt=.5):
    """Evaluator-only synthetic forecasts. Coverage is explicit, including next day."""
    origins=pd.date_range(day,periods=M,freq=pd.Timedelta(hours=dt))
    frames=[]
    for k,origin in enumerate(origins):
        valid=pd.date_range(origin+pd.Timedelta(hours=dt),periods=N-1,freq=pd.Timedelta(hours=dt))
        if not valid.isin(operation.index).all():raise ValueError('Actual-weather coverage is incomplete.')
        target=operation.loc[valid,'outdoor_temperature_c'].to_numpy(copy=True)
        forecast=target+case.forecast_bias_c+case.forecast_std_c*draws['forecast'][k]
        frames.append(pd.DataFrame({'issue_time':origin,'valid_time':valid,
            'lead_steps':np.arange(1,N),'outdoor_temperature_forecast_c':forecast}))
    return pd.concat(frames,ignore_index=True)


def controlled_plant(model,case,draws):
    """Adapter to the Day 1 reset/step contract; a single physical state."""
    model=dict(model)
    class ControlledReference:
        def __init__(self,parameters=None,step_hours=.5,random_state=11):
            self.parameters=parameters;self.step_hours=step_hours;self.position=0
            self.indoor_temperature_c=np.nan;self.fabric_temperature_c=np.nan
        def reset(self,indoor_temperature_c=20.,fabric_temperature_c=20.):
            self.indoor_temperature_c=float(indoor_temperature_c);self.position=0
            return self.observe()
        def observe(self):
            return float(self.indoor_temperature_c+case.sensor_bias_c+case.sensor_std_c*draws['sensor'][self.position])
        def step(self,heat_pump_modulation,outdoor_temperature_c,solar_irradiance_w_m2=0.,internal_gains_kw=0.):
            k=self.position
            power=float(np.clip(heat_pump_modulation,0.,1.)*model['heat_pump_max_electric_kw'])
            multiplier=case.loss_multiplier if k>=case.drift_after_step else 1.
            b_ref=model['loss_coefficient_per_step']*multiplier
            self.indoor_temperature_c=(1-b_ref)*self.indoor_temperature_c+b_ref*float(outdoor_temperature_c)+model['heat_coefficient_c_per_kw_step']*model['nominal_cop']*power+case.process_std_c*draws['process'][k]
            self.position+=1
            return {'indoor_temperature_measured_c':self.observe(),
                'indoor_temperature_true_c':float(self.indoor_temperature_c),
                'fabric_temperature_c':np.nan,'heat_pump_electric_kw':power,
                'delivered_heat_kw':power*model['nominal_cop'],'cop':model['nominal_cop'],'solar_gain_kw':0.}
    return ControlledReference


def run_case(operation,model,case_name,case,day,seed=11,strategies=('Fixed schedule','MPC'),margin=0.):
    validate_case(case,model)
    if margin<0 or (operation.comfort_min_c+margin>=operation.comfort_max_c-margin).any():
        raise ValueError('Margin must retain a nonempty comfort band.')
    draws=innovations(seed);forecast=forecast_table(operation,day,case,draws)
    plant=controlled_plant(model,case,draws);results={};metrics=[]
    for strategy in strategies:
        result=run_experiment(strategy,operation,forecast,model,day=day,M=48,N=48,
            seed=seed,gamma=25.,comfort_margin_c=margin,plant_class=plant)
        result['boundaries']=result['boundaries'].drop(columns=['fabric_temperature_c'])
        result.update(reference='Controlled one-state',case=case_name,date=day,seed=seed,
            case_settings=asdict(case),comfort_margin_c=margin,
            innovation_sha256={name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in draws.items()})
        rows=result['intervals'];rows['process_innovation']=draws['process'][:len(rows)]
        rows['sensor_innovation_start']=draws['sensor'][:len(rows)]
        rows['loss_coefficient_ref']=[model['loss_coefficient_per_step']*(case.loss_multiplier if k>=case.drift_after_step else 1.) for k in range(len(rows))]
        records=result['forecast_records']
        if len(records):
            records=records.copy();records['actual_outdoor_c']=operation.loc[pd.DatetimeIndex(records.valid_time),'outdoor_temperature_c'].to_numpy()
            records['actual_minus_forecast_c']=records.actual_outdoor_c-records.outdoor_input_c
            result['forecast_records']=records
        results[strategy]=result
        score=realised_metrics(result)
        metrics.append({'date':day,'case':case_name,'seed':seed,'margin_c':margin,**score})
    return pd.DataFrame(metrics),results


def batch(operation,model,names,dates,seeds=(11,22,33),margin=0.,strategies=('Fixed schedule','MPC')):
    frames=[];runs={}
    for day in dates:
        for name in names:
            case=CASES[name]
            for seed in (seeds if case.stochastic else seeds[:1]):
                frame,result=run_case(operation,model,name,case,day,seed,strategies,margin)
                frames.append(frame)
                for strategy,value in result.items():runs[(day,name,seed,strategy)]=value
    return pd.concat(frames,ignore_index=True),runs


def pairs(metrics):
    keys=['date','case','seed','margin_c']
    fixed=metrics.loc[metrics.strategy=='Fixed schedule'];mpc=metrics.loc[metrics.strategy=='MPC']
    merged=fixed.merge(mpc,on=keys,how='outer',suffixes=('_fixed','_mpc'),indicator=True)
    merged['valid_pair']=merged._merge.eq('both')&merged.complete_fixed.fillna(False).astype(bool)&merged.complete_mpc.fillna(False).astype(bool)
    for key in ['bill_gbp','total_degree_hours','peak_kw']:
        merged['delta_'+key]=(merged[key+'_mpc']-merged[key+'_fixed']).where(merged.valid_pair)
    return merged.drop(columns='_merge')


def sources(root):
    names=['course_utils/week8_control.py','course_utils/week8_errors.py',
        'course_utils/reference_building.py','data/reference_building/operation_conditions.csv',
        'data/reference_building/operation_forecasts.csv','data/reference_building/verified_controller_model.json',
        'scripts/build_week8_day2.py']
    return {name:hashlib.sha256((Path(root)/name).read_bytes()).hexdigest() for name in names}


def freeze_protocol(root,development_dates,final_dates):
    if set(development_dates)&set(final_dates):raise ValueError('Development and final dates overlap.')
    value={'development_dates':list(development_dates),'final_dates':list(final_dates),
        'final_cases':['C1 Forecast noise','C2 Forecast bias','C3 Model error','C7 Combined errors'],
        'seeds':[11,22,33],'N':48,'M':48,'gamma':25.,'margin_c':0.,
        'cases':{name:asdict(case) for name,case in CASES.items()},'source_sha256':sources(root)}
    return value


def verify_protocol(root,value):
    expected=freeze_protocol(root,value['development_dates'],value['final_dates'])
    if value!=expected:raise ValueError('The frozen cases, settings or sources changed.')
