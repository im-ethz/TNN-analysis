"""
Parse Garmin FIT files to CSV
Makes use of the library https://github.com/dtcooper/python-fitparse
Uses the following message types:
- file_id
- workout
- sport
- activity
- field_description
- hr_zone
- power_zone
- session
- training_file
- record
- device_info
- event
- lap
The following message types are ignored:
- file creator - doesn't contain anything interesting it seems
- device_settings - nothing relevant
- user_profile - we already know this stuff from trainingpeaks
- zones_target - contains threshold  power, threshold heart rate and max heartrate
"""
import fitparse
import argparse
import numpy as np
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, help="Name of the fitfile to convert")
parser.add_argument('-i', '--input', type=str, default='', help="Path to the directory where the fitfile is located")
parser.add_argument('-o', '--output', type=str, default='', help="Path to the directory where to save the csv")
parser.add_argument('-v', '--verbose', type=bool, default=False, help="Whether to give strings on progress to the command line")
args = parser.parse_args()

fname = args.filename.rstrip('.fit')


# ----------------- read data
if args.verbose:
	print("Reading in data ...")

fitfile = fitparse.FitFile(args.input + '/' + args.filename)

message_types = []
for m in fitfile.messages:
	try:
		message_types_m = m.mesg_type.name
	except:
		message_types_m = m.mesg_type
	message_types.append(message_types_m)
message_types = pd.DataFrame(message_types)[0]

data_nans = np.array(fitfile.messages)[message_types.isna()] # extract nan messages

message_types = message_types.unique().tolist()

data = {i: list(fitfile.get_messages(i)) for i in message_types}
data.update({None:data_nans})

df = dict()
# ----------------- info
if args.verbose:
	print("Creating info file ...")

df.update({'info': pd.Series(dtype=object)})

# file id
for message in data['file_id']:
	for field in message:
		df['info'].loc[field.name] = field.value
df['info'].index = pd.MultiIndex.from_product([["file_id"], df['info'].index])
message_types.remove('file_id')

# workout
if 'workout' in data:
	for message in data['workout']:
		for field in message:
			df['info'].loc['workout', field.name] = field.value
	message_types.remove('workout')

# sport
if 'sport' in data:
	for message in data['sport']:
		for field in message:
			df['info'].loc['sport', field.name] = field.value
	message_types.remove('sport')

# activity
if 'activity' in data:
	for message in data['activity']:
		for field in message:
			df['info'].loc['activity', field.name] = field.value
	message_types.remove('activity')

# field description
if 'field_description' in data:
	for message in data['field_description']:
		df['info'].loc['units', message.fields[3].value] = message.fields[4].value
	message_types.remove('field_description')

# hr_zone
if 'hr_zone' in data:
	for i, field in enumerate(data['hr_zone'][0].fields):
		if field.name == 'high_bpm':
			hr_zone_field = i
	df['info'].loc['hr_zone', data['hr_zone'][0].name+' [%s]'%data['hr_zone'][0].fields[hr_zone_field].units] = [message.fields[hr_zone_field].value for message in data['hr_zone']]
	message_types.remove('hr_zone')

# power_zone
if 'power_zone' in data:
	for i, field in enumerate(data['power_zone'][0].fields):
		if field.name == 'high_value':
			power_zone_field = i
	df['info'].loc['power_zone', data['power_zone'][0].name+' [%s]'%data['power_zone'][0].fields[power_zone_field].units] = [message.fields[power_zone_field].value for message in data['power_zone']]
	message_types.remove('power_zone')

# session
if 'session' in data:
	for message in data['session']:
		for field in message:
			df['info'].loc['session', field.name] = field.value
	message_types.remove('session')

# training file
if 'training_file' in data:
	for message in data['training_file']:
		for field in message:
			df['info'].loc['training_file', field.name] = field.value
	message_types.remove('training_file')


# ----------------- data
if args.verbose:
	print("Creating data files ...")

def unpack_messages(messages):
	df = pd.DataFrame()
	for i, message in enumerate(messages):
		for field in message.fields:
			df.loc[i,field.name] = field.value
	return df

# record
df.update({'data' : unpack_messages(data['record'])})
message_types.remove('record')

# None
try:
	df.update({'nan' : unpack_messages(data[None])})
	message_types.remove(None)
except ValueError:
	pass

# device
try:
	df.update({'device' : unpack_messages(data['device_info'])})
except ValueError:
	df.update({'device' : pd.DataFrame()})
	for i, message in enumerate(data['device_info']):
		for field in message.fields:
			try:
				df['device'].loc[i,field.name] = field.value
			except ValueError:
				continue
if "serial_number" in df['device'].columns:
	for i, item in enumerate(df['device'].serial_number.dropna().unique()):
		df_tmp = df['device'][df['device'].serial_number == item].dropna(axis=1).drop('timestamp', axis=1).drop_duplicates().iloc[0]
		df_tmp.index = pd.MultiIndex.from_product([["device_%i"%i], df_tmp.index])
		df['info'] = df['info'].append(df_tmp)
else:
	for i, item in df['device'].iterrows():
		item.index = pd.MultiIndex.from_product([["device_0"], item.index])
		df['info'] = df['info'].append(item)
message_types.remove('device_info')

# event
df.update({'startstop' : unpack_messages(data['event'])})
message_types.remove('event')

# laps
if 'laps' in data:
	df.update({'laps' : pd.DataFrame()})
	for i, message in enumerate(data['lap']):
		for field in message.fields:
			if type(field.value) != tuple:
				df['laps'].loc[i,field.name] = field.value
			else:
				try:
					df['laps'].at[i,field.name] = field.value
					break
				except:
					df['laps'][field.name] = df['laps'][field.name].astype(object)
					df['laps'].at[i,field.name] = field.value
					break
				else:
					break
	message_types.remove('lap')

# ----------------- save files
if args.verbose:
	print("Saving data ...")
	print("Message types not processed: ", *tuple(message_types))

for name, df_i in df.items():
	if not os.path.exists(args.output + '/' + name):
		os.mkdir(args.output + '/' + name)
	df_i.to_csv(args.output + '/' + name + '/' + fname + '_' + name + '.csv')