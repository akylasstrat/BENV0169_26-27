"""Supplied views of existing Week 8 logs; no control or simulation changes."""
import numpy as np
import pandas as pd


def demand_window(runs, schedule, dt=.5):
    """Declare the highest-tariff mask from the reporting schedule, then score."""
    price = schedule.import_price_gbp_per_kwh
    mask = price.eq(price.max())
    evidence = pd.DataFrame({'interval_start': schedule.index,
                             'price_gbp_per_kwh': price.to_numpy(),
                             'in_window': mask.to_numpy()})
    records = []
    for strategy, result in runs.items():
        rows = result['intervals']
        if not result['complete'] or list(rows.interval_start) != list(schedule.index):
            raise ValueError('Demand-window comparison requires aligned complete days.')
        power = rows.applied_power_kw.to_numpy()
        evidence[strategy + '_power_kw'] = power
        records.append(dict(strategy=strategy,
            window_energy_kwh=dt*power[mask.to_numpy()].sum(),
            off_window_energy_kwh=dt*power[~mask.to_numpy()].sum(),
            energy_kwh=dt*power.sum(),
            below_degree_hours=dt*np.maximum(rows.lower_c-rows.true_end_c,0).sum(),
            above_degree_hours=dt*np.maximum(rows.true_end_c-rows.upper_c,0).sum(),
            final_temperature_c=rows.true_end_c.iloc[-1]))
    return pd.DataFrame(records), evidence


def residual_evidence(result):
    """Use accepted stored first-step predictions, never a retrospective fit."""
    rows = result['intervals']
    keep = (rows.accepted & ~rows.fallback & rows.solve_attempted
            & rows.prediction_lead.eq(1)
            & np.isfinite(rows.predicted_next_c)
            & np.isfinite(rows.measured_minus_predicted_c))
    selected = rows.loc[keep].copy()
    states = result['boundaries'].set_index('state_time')
    selected['measured_minus_true_c'] = (
        states.loc[pd.DatetimeIndex(selected.interval_end),'measured_temperature_c'].to_numpy()
        - states.loc[pd.DatetimeIndex(selected.interval_end),'true_temperature_c'].to_numpy())
    return selected[['interval_start','interval_end','measured_start_c',
                     'measured_end_c','true_end_c','predicted_next_c',
                     'measured_minus_predicted_c','measured_minus_true_c']]


def case_evidence(paired):
    records=[]
    for case, group in paired.groupby('case',sort=False):
        valid=group.loc[group.valid_pair]
        records.append(dict(case=case,pairs=len(group),valid_pairs=len(valid),
            invalid_pairs=len(group)-len(valid),
            mpc_lower_bill=int((valid.delta_bill_gbp<0).sum()),
            mpc_lower_discomfort=int((valid.delta_total_degree_hours<0).sum()),
            mean_bill_difference_gbp=valid.delta_bill_gbp.mean(),
            mean_discomfort_difference_degree_hours=valid.delta_total_degree_hours.mean()))
    return pd.DataFrame(records)
