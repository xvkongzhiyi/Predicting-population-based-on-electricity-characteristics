
import argparse, numpy as np, pandas as pd
from datetime import datetime, timedelta
import math, random

def gen_user_series(start, hours, base_level, daily_amp, weekly_amp, temp_base, temp_amp, noise=0.1):
    ts = []
    cur = start
    for h in range(hours):
        # temperature: daily sinusoid + weekly drift
        day_frac = (h % 24)/24.0
        week_frac = (h % (24*7))/(24*7.0)
        temp = temp_base + temp_amp*math.sin(2*math.pi*day_frac) + 2.0*math.sin(2*math.pi*week_frac)
        # power y depends on temp (U-shape) + daily/weekly seasonality + base
        season_day = daily_amp*math.sin(2*math.pi*day_frac - math.pi/3)
        season_week = weekly_amp*math.sin(2*math.pi*week_frac)
        temp_effect = 0.02*(temp-22.0)**2
        y = base_level + season_day + season_week + temp_effect + random.gauss(0, noise)
        ts.append((cur, temp, y))
        cur += timedelta(hours=1)
    return ts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, default='data/power_sample.csv')
    ap.add_argument('--n_users', type=int, default=5)
    ap.add_argument('--days', type=int, default=14)
    args = ap.parse_args()

    start = datetime(2025,1,1,0,0,0)
    hours = args.days*24

    rows = []
    for u in range(args.n_users):
        base = 0.5 + 0.3*u
        daily = 0.8 + 0.1*u
        weekly = 0.5 + 0.05*u
        temp_base = 20 + u*0.5
        temp_amp = 6 + u*0.3
        series = gen_user_series(start, hours, base, daily, weekly, temp_base, temp_amp, noise=0.15)
        for ds, temp, y in series:
            rows.append({'user_id': u, 'ds': ds, 'temp': temp, 'y': y})
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print('Wrote', args.out, 'rows=', len(df))

if __name__ == '__main__':
    main()
