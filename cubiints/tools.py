import os
import numpy as np
import dask.array as da
import warnings
import line_profiler
profile = line_profiler.LineProfiler()

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


def __get_interval(rms, P, Na, SNR=3):

		# import ipdb; ipdb.set_trace()
		
		if np.isnan(P) or np.isnan(rms) or np.isnan(Na) or Na<2:
			return 0
		else:
			Nvis = int(np.ceil(SNR**2.*rms**2./((Na-1.)*P**2.)))
			return Nvis


get_interval = np.vectorize(__get_interval)


def define_time_chunks(timecol, size, scans, jump=1):
	"""
      	Construct the array indices for the different time chunks.

        Args:
            timecol (array): 
                measurement set time column.
            size (int): 
                chunk size.
            scans (array):   
                measurement scan numbers column.
           	jump (int, optional): 
                The magnitude of a jump has to be over this value to force a chunk boundary.             
        Returns:
            time_chunks:
                - list of chunks indices
	"""

	# import pdb; pdb.set_trace

	unique, tindex = np.unique(timecol, return_index=True)
	b = abs(np.roll(scans, 1) - scans) > jump
	bounds = np.append(np.where(b==True)[0], len(scans))

	rmap = {x: i for i, x in enumerate(unique)}
	indices  = np.fromiter(list(map(rmap.__getitem__, timecol)), int)

	time_chunks = []
	i=k=0

	while(i<len(unique)):
		if b[tindex[i]]:
			k +=1
		ts = tindex[i]
		if (indices[ts]+size)<len(tindex):
			if tindex[i+size] < bounds[k]:
				te = tindex[i+size]
				i = i + size
			else:
				te = bounds[k]
				i = indices[te+1]
			time_chunks.append((ts,te))
		else:
			te = bounds[-1]
			time_chunks.append((ts,te))
			# print("breaking from here", bounds[-1])
			break

	# print("i, k, after", i, k)

	LOGGER.info("Found {:d} time chunks from {:d} unique timeslots.".format(len(time_chunks), len(unique)))

	return time_chunks

def define_freq_chunks(size, nfreq):
	"""
		Contruct the array indices for the different frequency chunks

		Args:
			size (int):
				chunk size
			nfreq (int):
				number of frequencies

		Returns:
			freq_chunks:
				-list of chunks indices
	"""
	
	bounds = list(range(0,nfreq,size)) + [nfreq]

	freq_chunks = [(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)]

	LOGGER.info("Found {:d} frequency chunks from {:d} frequency channels.".format(len(freq_chunks), nfreq))

	return freq_chunks


def get_mean_n_ant_tf(ant1, ant2, f, time_chunks, freq_chunks, time_col, indices=None):
	"""
		Compute the mean number of antennas per time-frequency chunk

		Args:
			ant1 (1D array):
				antenna 1 values from measurement set
			ant2 (1D array):
				antenna 2 values from meausrement set
			f (3D array):
				flag column 
			time_chunks:
				list of time chunk indices
			freq_chunks:
				list of freq chunk indices
			time_col:
				time column from measurement set
			indices:
				tuple, use only these chunks


		returns
			n_ants (array):
				shape: 3 x len(time_chunks) x len(freq_chunks)
				3 for mean, min and max nmber of antennas
	
	"""

	n_ant = max(ant2) + 1
	
	# n_ants = np.zeros((3, tsize, fsize))

	nfreq = freq_chunks[0][1] - freq_chunks[0][0]

	
	if indices is None:
		tsize, fsize =  len(time_chunks), len(freq_chunks)

		rows, cols = np.indices((tsize, fsize))
		indices = list(zip(rows.flatten(), cols.flatten()))

		dim_1 = False

	else:
		tsize = fsize = len(indices)
		
		n_ants = np.zeros((len(indices), nfreq, n_ant))

		dim_1 = True

	# pdb.set_trace()

	def __antenna_worker(ind, loc):

		tind, find = ind

		tsel = slice(time_chunks[tind][0], time_chunks[tind][1])
		fsel = slice(freq_chunks[find][0], freq_chunks[find][1])

		times = time_col[tsel]
		nch = freq_chunks[find][1] - freq_chunks[find][0]

		Nt = float(len(np.unique(times)))
		fp=f[tsel, fsel, 0].squeeze()
		

		ant1p = ant1[tsel]
		ant2p = ant2[tsel]

		ant1p = 1+np.repeat(ant1p[:,np.newaxis], nch, axis=1)
		ant2p = 1+np.repeat(ant2p[:,np.newaxis], nch, axis=1)

		ant1p*=(fp==False)
		ant2p*=(fp==False)

		ant_counts = np.zeros((nch, n_ant))

		for fi in range(nch):

			c_ant1, count_1 = np.unique(ant1p[:, fi], return_counts=True)
			c_ant2, count_2 = np.unique(ant2p[:, fi], return_counts=True)

			for aa in range(1, n_ant+1, 1):
				try:
					caa1 = count_1[np.where(c_ant1==aa)][0]
				except:
					caa1 = 0
				try:
					caa2 = count_2[np.where(c_ant2==aa)][0]
				except:
					caa2 = 0

					
				ant_counts[fi, aa-1] = float(caa1 + caa2)

		# print ant_counts, "scan id = %d"%(scan_id)
		ant_zeros = np.array(range(1, n_ant+1, 1))[np.where(ant_counts==0)[1]]

		if any(ant_zeros):
			LOGGER.debug("Completely flagged antennas:[{}]".format(", ".join('{}'.format(col) for col in ant_zeros)) + " for chunk T%dF%d"%(tind,find))

		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=RuntimeWarning)
			ant_counts = np.where(ant_counts==0, np.nan, ant_counts)/Nt
			# n_ants[:,tind,find] = 1+np.nanmin(ant_counts), 1+np.nanmax(ant_counts), 1+np.nanmean(ant_counts)   #should be using nanmean

			if dim_1:
				n_ants[loc] = ant_counts + 1
			else:
				n_ants[tind, find] = ant_counts + 1

	for loc, index in enumerate(indices):
		__antenna_worker(index, loc)

		
	LOGGER.info("Done computing mean number of antennas")

	return n_ants


def get_flag_ratio(time_chunks, freq_chunks, f):
	"""
		compute the flag percentage in the different time tchunks

		Args:
			time_chunks (list)
				- list of tuple for the time chunk boundaries
			freq_chunks (list)
				- list of frequency tupple for the time chunk boundaries
			f (boolean array)
				-flag column from the measurement set
		returns:
			flags_ratio (array)
				-array containing the flag percentage of each time-frequency chunk
	"""

	tsize, fsize = len(time_chunks), len(freq_chunks)
	flags_ratio = np.zeros((tsize, fsize))

	for tt, time_chunk in enumerate(time_chunks):
		for ff, freq_chunk in enumerate(freq_chunks):
			tsel = slice(time_chunk[0], time_chunk[1])
			fsel = slice(freq_chunk[0], freq_chunk[1])

			f_chunk = f[tsel, fsel, 0]
			flags_ratio[tt,ff] = 100*np.sum(f_chunk)/f_chunk.size

	return flags_ratio


@profile
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
			raise RuntimeError("return_dask is only applies to columns without a Frequency axis, like WEIGHT sometimes")
		else:
			prealloc = prealloc[:, np.newaxis, :]
			_, nfreq, ncorr = prealloc.shape

			return da.from_array(prealloc, chunks=(row_chunks, nfreq, ncorr))

	if freqslice==slice(None):
		return prealloc
	else:
		return prealloc[:,freqslice,:]

@profile
def build_flag_colunm(tt, minbl=100, obvis=None, freqslice=slice(None), row_chunks=4000, cubi_flags=False):
	"""Construct the a the initial flag column that will be use by Cubical
	when flagset is set to -cubical and min base line of 100"""

	# import pdb; pdb.set_trace()

	uvw0 =  fetch("UVW", subset=tt)

	# TODO: Explain this and handle it better in the user options 

	try:
		bflagrow = fetch("BITFLAG_ROW", subset=tt)
		bflagcol = fetch("BITFLAG", subset=tt, freqslice=freqslice)
		bitf = tt.getcolkeyword("BITFLAG", "FLAGSET_legacy")
		cubif = tt.getcolkeyword("BITFLAG", "FLAGSET_cubical")
	except:
		LOGGER.info("No BITFLAG column in MS will default to FLAG/FLAG_ROW columns")
		bflagrow = fetch("FLAG_ROW", subset=tt)
		bflagcol = fetch("FLAG", subset=tt, freqslice=freqslice)

		try:
			bitf = tt.getcolkeyword("FLAG", "FLAGSET_legacy")
			cubif = tt.getcolkeyword("FLAG", "FLAGSET_cubical")
		except:
			bitf = 1
			cubif = 2

	
	flag0 = np.zeros_like(bflagcol)

	#flag the with baseline length
	if minbl:
		uv2 = (uvw0[:, 0:2] ** 2).sum(1)
		flag0[uv2 < minbl**2] = 1
		del uvw0, uv2

	# Exceptionally add CubiCal flags
	if cubi_flags:
		apply_bit = bitf | cubif
	else:
		apply_bit = bitf

	#bitflag column
	flag0[(bflagcol & apply_bit) != 0] = True

	#bitflag row
	flag0[(bflagrow & apply_bit) != 0, :, :] = True

	#remove invalid visibilities
	if type(obvis) is np.ndarray:
		ax = obvis[...,(0,-1)] == 0
		flag0[ax[:,:,0]==True] = 1
		del ax


	percent_flag = 100.*np.sum(flag0[:,:,0])/flag0[:,:,0].size
	
	LOGGER.info("Flags fraction {}".format(percent_flag))

	del bflagrow, bflagcol

	_, nfreq, ncorr = flag0.shape

	flag0 = da.from_array(flag0, chunks=(row_chunks, nfreq, ncorr))

	return flag0
