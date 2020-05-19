import os
import numpy as np
import dask.array as da

import logging

LOGGER = logging.getLogger(__name__)

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

	return log

LOGGER = create_logger(__name__)

def get_scan_bounds(scans, jump=0):
	"""return the index bounds of the different scans"""

	b = abs(np.roll(scans, 1) - scans) > jump
	bounds = np.append(np.where(b==True)[0], -1)
	
	if len(bounds) == 1:
		scan_ids = np.array(range(len(bounds)), dtype=int)
		bounds = np.insert(bounds, 0, 0)
	else:
		scan_ids = np.array(range(len(bounds)-1), dtype=int)

	
	return scan_ids, bounds


def get_mean_n_ant_scan(ant1, ant2, f, scans, time_col, jump=0):
	"""get number of antennas per antenna

	f is the flag table
	"""

	# import pdb; pdb.set_trace()
	
	scan_ids, bounds =  get_scan_bounds(scans, jump)
	
	nch = f.shape[1]
	n_ant = max(ant2) + 1
	n_ants = np.zeros(len(scan_ids))
	for i, scan_id in enumerate(scan_ids):

		if bounds[i+1] == -1:
			sel = slice(bounds[i], None)
		else:
			sel = slice(bounds[i], bounds[i+1])
		
		times = time_col[sel]
		Nt = float(len(np.unique(times)))*nch
		fp=f[sel][:,:, 0].squeeze()
		ant1p = ant1[sel] 
		ant2p = ant2[sel]
		
		ant1p = 1+np.repeat(ant1p[:,np.newaxis], nch, axis=1)
		ant2p = 1+np.repeat(ant2p[:,np.newaxis], nch, axis=1)

		ant1p*=(fp==False)
		ant2p*=(fp==False)

		ant_counts = np.zeros(n_ant)
		c_ant1, count_1 = np.unique(ant1p, return_counts=True)
		c_ant2, count_2 = np.unique(ant2p, return_counts=True)

		for aa in range(1, n_ant+1, 1):
			try:
				caa1 = count_1[np.where(c_ant1==aa)][0]
			except:
				caa1 = 0
			try:
				caa2 = count_2[np.where(c_ant2==aa)][0]
			except:
				caa2 = 0

				
			ant_counts[aa-1] = float(caa1 + caa2)

		# print ant_counts, "scan id = %d"%(scan_id)
		# report if any is zero
		ant_zeros = np.array(range(1, n_ant+1, 1))[np.where(ant_counts==0)]

		if any(ant_zeros):
			LOGGER.info("Completely flagged antennas:[{}]".format(", ".join('{}'.format(col) for col in columns)) + " for scan %d"%scan_id)

		# print(np.array(range(1, n_ant+1, 1))[np.where(ant_counts==0)], " scan id = %d"%scan_id)

		ant_counts = np.where(ant_counts==0, np.nan, ant_counts)

		# add one for the antenna itself

		n_ants[i] = np.nanmin(ant_counts)/Nt + 1 #should be using nanmean

	return n_ants


def fetch(colname, first_row=0, nrows=-1, subset=None, freqslice=slice(None), dtype=None, row_chunks=4000, return_dask=False):
	"""
	Convenience function which mimics pyrap.tables.table.getcol().
	Args:
	    colname (str):
	        column name
	    first_row (int):
	        starting row
	    nrows (int):
	        number of rows to fetch
	    subset:
	        table to fetch from, else uses self.data
	Returns:
	    np.ndarray:
	        Result of getcol(\*args, \*\*kwargs).

	"""

	# import pdb; pdb.set_trace()

	# ugly hack because getcell returns a different dtype to getcol
	cell = subset.getcol(str(colname), first_row, nrow=1)[0, ...]
	if dtype is None:
		dtype = getattr(cell, "dtype", type(cell))
	nrows = subset.nrows() if nrows < 0 else nrows
	shape = tuple([nrows] + [s for s in cell.shape]) if hasattr(cell, "shape") else nrows
	prealloc = np.empty(shape, dtype=dtype)
	subset.getcolnp(str(colname), prealloc, first_row, nrows)

	if return_dask:
		if len(prealloc.shape) > 2:
			raise RuntimeError("return_dask is only applies to columns without a Frequency axis, like WEIGHT some times")
		else:
			prealloc = prealloc[:, np.newaxis, :]
			_, nfreq, ncorr = prealloc.shape

			return da.from_array(prealloc, chunks=(row_chunks, nfreq, ncorr))

	if freqslice==slice(None):
		return prealloc
	else:
		return prealloc[:,freqslice,:]


def build_flag_colunm(tt, minbl=100, bitf=1, obvis=None, freqslice=slice(None), row_chunks=4000, cubicalf=2):
	"""Construct the a the initial flag column that will be use by Cubical
	when flagset is set to -cubical and min base line of 100"""

	# import pdb; pdb.set_trace()

	uvw0 =  fetch("UVW", subset=tt)

	# TODO: Explain this and handle it better in the user options 

	try:
		bflagrow = fetch("BITFLAG_ROW", subset=tt)
		bflagcol = fetch("BITFLAG", subset=tt, freqslice=freqslice)
	except:
		LOGGER.info("No BITFLAG column in MS will default to FLAG/FLAG_ROW columns")
		bflagrow = fetch("FLAG_ROW", subset=tt)
		bflagcol = fetch("FLAG", subset=tt, freqslice=freqslice)

	
	flag0 = np.zeros_like(bflagcol)

	#flag the with baseline length
	uv2 = (uvw0[:, 0:2] ** 2).sum(1)
	flag0[uv2 < minbl**2] = 1

	#bitflag column
	flag0[(bflagcol & bitf) != 0] = 1

	#bitflag row
	flag0[(bflagrow & bitf) != 0, :, :] = 1

	# Exceptionally add CubiCal flags
	# if cubicalf:
	# 	flag0[(bflagcol & cubicalf) != 0] = 1
	# 	flag0[(bflagrow & cubicalf) != 0, :, :] = 1


	#remove invalid visibilities
	if type(obvis) is np.ndarray:
		ax = obvis[...,(0,-1)] == 0
		flag0[ax[:,:,0]==True] = 1
		del ax


	percent_flag = 100.*np.sum(flag0)/flag0.size
	
	LOGGER.info("Flags fraction {}".format(percent_flag))

	del bflagrow, bflagcol, uvw0, uv2

	_, nfreq, ncorr = flag0.shape

	flag0 = da.from_array(flag0, chunks=(row_chunks, nfreq, ncorr))

	return flag0