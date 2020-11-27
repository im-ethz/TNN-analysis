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

for message in data['file_id']:
	for field in message:
		df_info.loc[field.name] = field.value
df_info.index = pd.MultiIndex.from_product([["file_id"], df_info.index])
message_types.remove('file_id')

for message in data['workout']:
	for field in message:
		df_info.loc['workout', field.name] = field.value
message_types.remove('workout')

for message in data['sport']:
	for field in message:
		df_info.loc['sport', field.name] = field.value
message_types.remove('sport')

for message in data['activity']:
	for field in message:
		df_info.loc['activity', field.name] = field.value
message_types.remove('activity')

for message in data['field_description']:
	df_info.loc['units', message.fields[3].value] = message.fields[4].value
message_types.remove('field_description')

for i, field in enumerate(data['hr_zone'][0].fields):
	if field.name == 'high_bpm':
		hr_zone_field = i
df_info.loc['hr_zone', data['hr_zone'][0].name+' [%s]'%data['hr_zone'][0].fields[hr_zone_field].units] = [message.fields[hr_zone_field].value for message in data['hr_zone']]
message_types.remove('hr_zone')

for i, field in enumerate(data['power_zone'][0].fields):
	if field.name == 'high_value':
		power_zone_field = i
df_info.loc['power_zone', data['power_zone'][0].name+' [%s]'%data['power_zone'][0].fields[power_zone_field].units] = [message.fields[power_zone_field].value for message in data['power_zone']]
message_types.remove('power_zone')

for message in data['session']:
	for field in message:
		df_info.loc['session', field.name] = field.value
message_types.remove('session')


# ----------------- data
if args.verbose:
	print("Creating data files ...")

def unpack_messages(messages):
	df = pd.DataFrame()
	for i, message in enumerate(messages):
		for field in message.fields:
			df.loc[i,field.name] = field.value
	return df

df_data = unpack_messages(data['record'])
message_types.remove('record')

df_nan = unpack_messages(data[None])
message_types.remove(None)

df_device = unpack_messages(data['device_info'])
message_types.remove('device_info')
for i, item in enumerate(df_device.serial_number.dropna().unique()):
	df_tmp = df_device[df_device.serial_number == item].dropna(axis=1).drop('timestamp', axis=1).drop_duplicates().iloc[0]
	df_tmp.index = pd.MultiIndex.from_product([["device_%i"%i], df_tmp.index])
	df_info = df_info.append(df_tmp)

df_startstop = unpack_messages(data['event'])
message_types.remove('event')

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
		 'nan'		: df_nan,
		 'device'	: df_device,
		 'startstop': df_startstop,
		 'laps'		: df_laps}

for name, df in files.items():
	if not os.path.exists(args.output + '/' + name):
		os.mkdir(args.output + '/' + name)
	df.to_csv(args.output + '/' + name + '/' + fname + '_' + name + '.csv')