import numpy as np
import pandas as pd

import os
import gc

import datetime

from plot import *
from helper import *
from calc import *

from config import rider_mapping

path = 'Data/TrainingPeaks/2019/'
savepath = 'Descriptives/trainingpeaks/'

if not os.path.exists(savepath):
	os.mkdir(savepath)

athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(path+'clean/')])

# plot devices used
df_dev = {}
for i in athletes:
	print("\n------------------------------- Athlete ", i)

	df = pd.read_csv(path+'clean/'+str(i)+'/'+str(i)+'_data.csv', index_col=0)

	df['timestamp'] = pd.to_datetime(df['timestamp'])
	df['date'] = df['timestamp'].dt.date

	#df_dev[i] = df.groupby('date')['device_0'].unique()
	df_dev[i] = df.groupby(['device_0', 'date'])['device_0'].nunique()
	del df ; gc.collect()

df_dev = pd.concat(df_dev).unstack().unstack().fillna(0)

colors = ('Reds', 'Oranges', 'Greens', 'Blues', 'Purples')

fig, ax = plt.subplots(figsize=(15,6))
#for j, dev in enumerate(df_dev.columns.get_level_values(1).unique()):
sns.heatmap(df_dev.xs(dev, axis=1, level=1), ax=ax, mask=df_dev.xs(dev, axis=1, level=1) == 0,
		cmap=colors[j], cbar=False, linewidths=.01, vmin=1, vmax=1, center=1)
plt.xticks(ticks=[d+15 for d in month_firstday.values()], labels=[list(month_firstday.keys())[-1]]+list(month_firstday.keys())[:-1], rotation=0)
plt.ylabel('rider')

plt.savefig(savepath+'device_usage_day_all.pdf', bbox_inches='tight')
plt.savefig(savepath+'device_usage_day_all.png', dpi=300, bbox_inches='tight')
ax.set_yticklabels([rider_mapping_inv[int(i.get_text())] for i in ax.get_yticklabels()], rotation=0)
plt.savefig(savepath+'device_usage_day_all_NAME.pdf', bbox_inches='tight')
plt.savefig(savepath+'device_usage_day_all_NAME.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()