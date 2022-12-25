import pandas as pd
df = pd.read_csv('/Users/emma/dev/TurboGrad/year_of_registration.csv')
vals=df['year_of_registration'].values.tolist()

print(len(vals))
print(type(vals[50]))
print(vals[:50])
n_nan =0
for date in vals:
    if date == float('nan'):
        n_nan +=1

print(n_nan)