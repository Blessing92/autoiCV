# -*- coding: utf-8 -*-
from memory_profiler import memory_usage
import os
import sys

def StartMeMonitoring(pid, datasetName):
    filename =  datasetName + '-memory.txt'
    print("Memory Logger Start\n")
    print("Result File: " + filename)
    """
	if not os.path.exists(os.path.dirname(filename)):
		try:
			os.makedirs(os.path.dirname(filename))
		except OSError as exc: # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise
	"""
    f = open(filename, "a")
    mem_usage = memory_usage(int(pid), interval=1, timeout=3600000, retval=False, stream=f)
    
if __name__ == '__main__':
    
    StartMeMonitoring(sys.argv[1], sys.argv[2])
    #StartMeMonitoring(os.getpid(), 'a7a')