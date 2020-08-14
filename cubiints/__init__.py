import os
import logging

log_init = False

def create_logger(name):
	"""Create a console logger"""
	log = logging.getLogger(name)
	cfmt = logging.Formatter(('%(module)s - %(asctime)s %(levelname)s - %(message)s'))
	log.setLevel(logging.DEBUG)
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	console.setFormatter(cfmt)
	log.addHandler(console)

	logfile = 'log-interval.txt'

	global log_init

	if not log_init:
		if os.path.isfile(logfile):
			import glob
			nb_runs = len(glob.glob(logfile+"*"))

			import shutil
			shutil.move(logfile, logfile+"-"+str(nb_runs-1))

		log_init = True

	fh = logging.FileHandler(logfile)
	fh.setLevel(logging.INFO)
	fh.setFormatter(cfmt)
	log.addHandler(fh)

	log.propagate = False

	return log

LOGGER = create_logger("cubiints")