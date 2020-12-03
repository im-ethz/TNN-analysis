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

# ----------------- info
if args.verbose:
	print("Creating info file ...")

df_info = pd.Series(dtype=object)

# file id
for message in data['file_id']:
	for field in message:
		df_info.loc[field.name] = field.value
df_info.index = pd.MultiIndex.from_product([["file_id"], df_info.index])
message_types.remove('file_id')

# workout
try:
	for message in data['workout']:
		for field in message:
			df_info.loc['workout', field.name] = field.value
	message_types.remove('workout')
except KeyError:
	pass

# sport
try:
	for message in data['sport']:
		for field in message:
			df_info.loc['sport', field.name] = field.value
	message_types.remove('sport')
except KeyError:
	pass

# activity
for message in data['activity']:
	for field in message:
		df_info.loc['activity', field.name] = field.value
message_types.remove('activity')

# field description
try:
	for message in data['field_description']:
		df_info.loc['units', message.fields[3].value] = message.fields[4].value
	message_types.remove('field_description')
except KeyError:
	pass

# hr_zone
try:
	for i, field in enumerate(data['hr_zone'][0].fields):
		if field.name == 'high_bpm':
			hr_zone_field = i
	df_info.loc['hr_zone', data['hr_zone'][0].name+' [%s]'%data['hr_zone'][0].fields[hr_zone_field].units] = [message.fields[hr_zone_field].value for message in data['hr_zone']]
	message_types.remove('hr_zone')
except KeyError:
	pass

# power_zone
try:
	for i, field in enumerate(data['power_zone'][0].fields):
		if field.name == 'high_value':
			power_zone_field = i
	df_info.loc['power_zone', data['power_zone'][0].name+' [%s]'%data['power_zone'][0].fields[power_zone_field].units] = [message.fields[power_zone_field].value for message in data['power_zone']]
	message_types.remove('power_zone')
except KeyError:
	pass

# session
for message in data['session']:
	for field in message:
		df_info.loc['session', field.name] = field.value
message_types.remove('session')

# training file
try:
	for message in data['training_file']:
		for field in message:
			df_info.loc['training_file', field.name] = field.value
	message_types.remove('training_file')
except KeyError:
	pass

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
df_data = unpack_messages(data['record'])
message_types.remove('record')

# None
try:
	df_nan = unpack_messages(data[None])
	message_types.remove(None)
except ValueError:
	pass

# device
try:
	df_device = unpack_messages(data['device_info'])
except ValueError:
	df_device = pd.DataFrame()
	for i, message in enumerate(data['device_info']):
		for field in message.fields:
			try:
				df_device.loc[i,field.name] = field.value
			except ValueError:
				continue
if "serial_number" in df_device.columns:
	for i, item in enumerate(df_device.serial_number.dropna().unique()):
		df_tmp = df_device[df_device.serial_number == item].dropna(axis=1).drop('timestamp', axis=1).drop_duplicates().iloc[0]
		df_tmp.index = pd.MultiIndex.from_product([["device_%i"%i], df_tmp.index])
		df_info = df_info.append(df_tmp)
else:
	for i, item in df_device.iterrows():
		item.index = pd.MultiIndex.from_product([["device_0"], item.index])
		df_info = df_info.append(item)
message_types.remove('device_info')

# event
df_startstop = unpack_messages(data['event'])
message_types.remove('event')

# laps
df_laps = pd.DataFrame()
for i, message in enumerate(data['lap']):
	for field in message.fields:
		if type(field.value) != tuple:
			df_laps.loc[i,field.name] = field.value
		else:
			try:
				df_laps.at[i,field.name] = field.value
				break
			except:
				df_laps[field.name] = df_laps[field.name].astype(object)
				df_laps.at[i,field.name] = field.value
				break
			else:
				break
message_types.remove('lap')

# ----------------- save files
if args.verbose:
	print("Saving data ...")
	print("Message types not processed: ", *tuple(message_types))

files = {'info'		: df_info,
		 'data'		: df_data,
		 'device'	: df_device,
		 'startstop': df_startstop,
		 'laps'		: df_laps}

for name, df in files.items():
	if not os.path.exists(args.output + '/' + name):
		os.mkdir(args.output + '/' + name)
	df.to_csv(args.output + '/' + name + '/' + fname + '_' + name + '.csv')

try:
	if not os.path.exists(args.output + '/nan'):
		os.mkdir(args.output + '/nan')	
	df_nan.to_csv(args.output + '/nan/' + fname + '_nan.csv')
except NameError:
	pass