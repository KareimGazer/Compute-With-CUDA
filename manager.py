#!/usr/bin/env python3

# installed by default in python3
import csv
import os
import time

raw_data_file = "processes.txt"

# return a dictionary {PID: VSZ} for all processes
def get_proc_mem(raw_data_file):
	# file to store processes data after cleaning
	cleanProcData = open("procs.csv", 'w')
	# processing the data
	with open(raw_data_file, newline='\n') as csvfile:
		for row in csvfile:
			writer = csv.writer(cleanProcData)
			writer.writerow(row.split())
	cleanProcData.close()
	
	# getting the needed info
	with open("procs.csv", newline='\n') as csvfile:
		file_read = csv.reader(csvfile)
		info_table = list(file_read)
		PID = [row[1] for row in info_table]
		VSZ = [row[4] for row in info_table]
	result =  dict(zip(PID, VSZ))
	del result["PID"]
	return result
	
def scan():
	# using shell command to get processes info into a file
	os.system("ps -aux >> {}".format(raw_data_file))
	# getting the data formatted in dictionary
	info1 = get_proc_mem(raw_data_file)
	time.sleep(10)
	os.system("ps -aux >> {}".format(raw_data_file))
	info2 = get_proc_mem(raw_data_file)

	# measuring the memory consumption for each process in MB
	results = dict()
	for pid in info2:
		diff = int(info2.get(pid, 0)) - int(info1.get(pid, 0))
		results[pid] = abs(round(diff / 1024, 1)) # return MB

	black_list = list() # list of suspicious processes
	for key in results:
		if results[key] == 200.0:
			black_list.append(key)
	return black_list

# detects the program by scanning multiple times
# to make sure the memory usage is periodic
def detect(iterations_num=2):
	# using sets to detect repeated processes
	suspicious = set()
	for i in range(iterations_num):
		iter_result = set(scan())
		print("{}: detected processes {}".format(i, iter_result))
		if i == 0:
			suspicious = iter_result
		else:
			suspicious = suspicious.intersection(iter_result)
	return list(suspicious)
	
malware_list = detect()

for pid in malware_list:
	print("found {} and shutting it down".format(pid))
	os.system("kill {}".format(pid))



