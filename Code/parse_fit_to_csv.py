import fitparse
import argparse
import numpy as np
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument('f')

fname = "2020-09-19-084502-ELEMNT BOLT B385-222-0.fit"

fitfile = fitparse.FitFile(fname)

message_types = ("record", "device_info", "developer_data_id", "field_description", "event", "power_zone", "hr_zone", "lap", "activity", "sport", "workout", "file_id", "session")

data = {i: list(fitfile.get_messages(i)) for i in message_types}

def unpack_messages(messages):
	df = pd.DataFrame()
	for i, message in enumerate(messages):
		for field in message.fields:
			df.loc[i,field.name] = field.value
	return df

# header
#TODO: lap, activity, session
df_info = {}
for message in data['file_id']:
	for field in message:
		df_info.update({field.name:field.value})
df_info.update({data['workout'][0].name:data['workout'][0].fields[0].value})
df_info.update({data['sport'][0].name:data['sport'][0].fields[0].value})
for message in data['field_description']:
	df_info.update({message.fields[3].value:message.fields[4].value})
df_info.update({data['hr_zone'][0].name+' [%s]'%data['hr_zone'][0].fields[1].units:[message.fields[1].value for message in data['hr_zone']]})
df_info.update({data['power_zone'][0].name+' [%s]'%data['power_zone'][0].fields[1].units:[message.fields[1].value for message in data['power_zone']]})
for message in data['event']:
	[[field.value for field in message] for message in data['event']]



# data
df_data = unpack_messages(data['record'])
df.to_csv(fname.rstrip('.fit'))

df_device = unpack_messages(data['device_info'])

df_startstop = unpack_messages(data['event'])

df_laps = pd.DataFrame()
for i, message in enumerate(messages):
	for field in message.fields:
		print(field.name)
		if type(field.value) != tuple:
			df_laps.loc[i,field.name] = field.value
		else:
			print("field is tuple")
			try:
				print("try")
				df_laps.at[i,field.name] = field.value
				break
				break
				break
			except:
				print("except")
				df_laps[field.name] = df_laps[field.name].astype(object)
				df_laps.at[i,field.name] = field.value
				break
			else:
				break
			#print("field is not tuple")

	#		except ValueError:
#			df_laps[field.name] = df_laps[field.name].astype(object)
#			df_laps.at[i,field.name] = field.value


unpack_messages(data['lap'])

"""
messages = fitfile.messages

header = {}
for field in messages[0].fields:
	header.update({field.name:field.value})

fitfile.close()

types = []
for m in messages:
	types.append(m.type)
types = np.array(types)
print(np.unique(types))

message_types = []
for m in messages:
	try:
		message_types_m = m.mesg_type.name
	except:
		message_types_m = m.mesg_type
	message_types.append(message_types_m)
message_types = pd.DataFrame(message_types)
print(message_types[0].value_counts())
"""

