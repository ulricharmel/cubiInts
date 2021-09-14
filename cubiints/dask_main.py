import argparse
from pyrap.tables import table
import numpy as np
import Tigger

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import os

import traceback

try:
	import ipdb as pdb
except:
	import pdb

import warnings

from matplotlib import rcParams 
#import aplpy
rcParams.update({'font.size': 20, 'font.family': 'sans-serif'})
plt.rcParams["figure.figsize"] = [16,9]
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

import time
import logging 
mpl_logger = logging.getLogger('matplotlib') 
mpl_logger.setLevel(logging.WARNING)


#------dask ms -----stuffs-----------------------#
from daskms import xds_from_ms, xds_from_table
import dask
import dask.array as da
from cubiints.tools import *
from cubiints.aic_functions import optimal_time_freq_interval_from_gains, optimal_time_freq_interval_2D_gains, optimal_time_freq_interval_2D_same_gains

xds = []

# LOGGER = create_logger(__name__)
from cubiints import LOGGER

@profile
def model_flux_per_scan_dask(time_chunks, freq_chunks, fluxcols, w, f, filename="M1", outdir="./soln-intervals", indices=None, tigger_model=False):
	"""compute the flux per interval scans"""

	if tigger_model:
		if indices is None:
			nt, nv = len(time_chunks), len(freq_chunks)
			flux = np.ones((nt, nv))
		else:
			flux = np.ones(len(indices))

		modelname = fluxcols[0]

		if "@" in modelname:
			modelname, tag = modelname.split("@")
			de = {}
			model = Tigger.load(modelname)
			for src in model.sources:
				if src.getTag(tag):
					de[src.cluster] = src.cluster_flux

			flux *= np.min(np.array(list(de.values()))) # min
		else:
			model = Tigger.load(modelname)
			clusters = {}
			for src in model.sources:
				try:
					clusters[src.cluster] = src.cluster_flux
				except AttributeError:
					clusters[src.name] = src.flux.I

			flux *= max(clusters.values())

		return flux


	else:
		if len(fluxcols) == 1:
			m0 = getattr(xds[0], fluxcols[0]).data
			__sub_model = False
		else:
			m1 = getattr(xds[0], fluxcols[0]).data
			m0 = getattr(xds[0], fluxcols[1]).data
			__sub_model = True


		# apply flags and weights
		
		m0*=(f==False)
		m0*=w

		if __sub_model:
			# p*=(f==False) select based on m only
			m1*=w
			
		LOGGER.debug("Done applying weights and flags")

		if indices is None:

			nt, nv = len(time_chunks), len(freq_chunks)

			flux = np.zeros((nt, nv))

			for tt, time_chunk in enumerate(time_chunks):
				for ff, freq_chunk in enumerate(freq_chunks):
					tsel = slice(time_chunk[0], time_chunk[1])
					fsel = slice(freq_chunk[0], freq_chunk[1])

					if __sub_model:
						model_abs = da.absolute(m1[tsel, fsel, :][...,[0,3]][m0[tsel, fsel, :][...,[0,3]]!=0] - m0[tsel, fsel,:][...,[0,3]][m0[tsel, fsel, :][...,[0,3]]!=0])
					else:
						model_abs = da.absolute(m0[tsel, fsel, :][...,[0,3]][m0[tsel, fsel, :][...,[0,3]]!=0])
					
					with warnings.catch_warnings():
						warnings.simplefilter("ignore", category=RuntimeWarning)
						flux[tt,ff] = np.mean(model_abs.compute())

				if tt%6 == 0:
					LOGGER.info("Done computing model flux for {%d}/{%d} time chunks"%(tt+1, len(time_chunks)))

			LOGGER.info("Done computing model flux")

			np.save(outdir+"/"+filename+"flux.npy", flux)

			return flux

		else:

			flux = np.zeros(len(indices))

			for loc, index in enumerate(indices):
				tsel = slice(time_chunks[index[0]][0], time_chunks[index[0]][1])
				fsel = slice(freq_chunks[index[1]][0], freq_chunks[index[1]][1])

				if __sub_model:
					model_abs = da.absolute(m1[tsel, fsel, :][...,[0,3]][m0[tsel, fsel, :][...,[0,3]]!=0] - m0[tsel, fsel,:][...,[0,3]][m0[tsel, fsel, :][...,[0,3]]!=0])
				else:
					model_abs = da.absolute(m0[tsel, fsel, :][...,[0,3]][m0[tsel, fsel, :][...,[0,3]]!=0])
					
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", category=RuntimeWarning)
					flux[loc] = np.mean(model_abs.compute())

			LOGGER.info("Done computing model flux")

			return flux

@profile
def compute_interval_dask_index(ms_opts={}, SNR=3, dvis=False, outdir="./soln-intervals", figname="interval", minbl=0, row_chunks=4000, freqslice=slice(None), jump=1, 
										tchunk=64, fchunk=128, save_out=True, cubi_flags=False, datachunk=None, allargs=None):
	"""replicate the compute interval using dask arrays"""

	t0 = time.time()

	t = table(ms_opts["msname"])

	f = build_flag_colunm(t, minbl=minbl, obvis=None, freqslice=freqslice, cubi_flags=cubi_flags)
	
	LOGGER.info("finished building flags")

	LOGGER.debug("Took {} seconds to complete".format(time.time()-t0))

	cell = t.getcell("DATA", rownr=1)
	nfreq = cell.shape[0]

	w = fetch(ms_opts["WeightCol"], subset=t, return_dask=True)
	LOGGER.info("read weight-column susscessful")

	t.close()
	LOGGER.info("Table Closed")

	if "lsm.html" in ms_opts["ModelCol"]:
		LOGGER.info("Tigger skymodel pass as model, will compute the noise from the data column only")
		columns = ["ANTENNA1", "ANTENNA2", "TIME", "SCAN_NUMBER", ms_opts["DataCol"]]  #,
		_model_col = False
	else:
		cols = ms_opts["FluxCol"].split("-")
		columns = list(set(["ANTENNA1", "ANTENNA2", "TIME", "SCAN_NUMBER", ms_opts["DataCol"], ms_opts["ModelCol"]]) | set(cols))
		_model_col = True

	LOGGER.info("Reading the columns: [{}] as a daskms".format(", ".join('{}'.format(col) for col in columns)))

	global xds

	xds = xds_from_ms(ms_opts["msname"], columns=columns, chunks={"row":row_chunks})

	scans = getattr(xds[0], "SCAN_NUMBER").data.compute()
	time_col = getattr(xds[0], "TIME").data.compute()
	ant1 = getattr(xds[0], "ANTENNA1").data.compute()
	ant2 = getattr(xds[0], "ANTENNA2").data.compute()

	NUMBER_ANTENNAS = max(ant2) + 1

	time_chunks = define_time_chunks(time_col, tchunk, scans, jump=jump)

	freq_chunks = define_freq_chunks(fchunk, nfreq)

	time_f = time.time()

	LOGGER.info("Done defining chunks and meta data-- Now to the expensive part")

	if datachunk:
		indices = [tuple(np.array(datachunk.split("T")[1].split("F"), dtype=int))]

	else:
		flags_ratio = get_flag_ratio(time_chunks, freq_chunks, f.compute())

		fr = np.nanmedian(flags_ratio)
		indices = [np.unravel_index(np.nanargmin(np.abs(flags_ratio-fr)), flags_ratio.shape)]

	n_ants = get_mean_n_ant_tf(ant1, ant2, f.compute(), time_chunks, freq_chunks, time_col, indices=indices) #[2,...]  # return only the min values in each chunk so out put is 2D 

	# print(np.where(n_ants<20), "see where n ants is less than 20 the most")

	n_ants[n_ants==0] = np.nan

	LOGGER.debug("Took {} seconds to compute flag ratio and antennas".format(time.time()-time_f))

	if dvis:
		raise NotImplementedError("Compute rms from visibilities only yet to be implemented")


	d = getattr(xds[0], ms_opts["DataCol"]).data
	d*=(f==False)
	
	if _model_col:
		p = getattr(xds[0], ms_opts["ModelCol"]).data
		p*=w
		d*=w

	LOGGER.info("Done applying weights and flags")

	LOGGER.info(f"Computing rms from data only set to {allargs.data_rms}")

	_prefix = figname.split("interval")[0]+"T"+str(indices[0][0])+"F"+str(indices[0][1])+"-"

	time_m = time.time()
	if _model_col:
		flux = model_flux_per_scan_dask(time_chunks, freq_chunks, cols, w, f, filename=_prefix, outdir=outdir, indices=indices)
	else:
		flux = model_flux_per_scan_dask(time_chunks, freq_chunks, [ms_opts["ModelCol"]], w, f, filename=_prefix, outdir=outdir, indices=indices, tigger_model=True)

	LOGGER.debug("Took {} seconds to compute model".format(time.time()-time_m))

	chan_rms = np.zeros((n_ants.shape[1], NUMBER_ANTENNAS))
	
	nv_nt = np.zeros_like(n_ants)

	grms_na = np.zeros_like(n_ants)

	time_c = time.time()

	for loc, index in enumerate(indices): 

		tsel = slice(time_chunks[index[0]][0], time_chunks[index[0]][1])
		fsel = slice(freq_chunks[index[1]][0], freq_chunks[index[1]][1])

		ant1p = ant1[tsel] 
		ant2p = ant2[tsel]

		# pdb.set_trace()

		for aa in range(NUMBER_ANTENNAS):			
			dps = d[tsel][(ant1p==aa)|(ant2p==aa)][:, fsel, :][...,[0,3]] #[d[(ant1p==aa)|(ant1p==aa)][:, fsel, :][...,[0,3]]!=0]

			if not allargs.data_rms:
				pps = p[tsel][(ant1p==aa)|(ant2p==aa)][:, fsel, :][...,[0,3]] #[d[(ant1p==aa)|(ant1p==aa)][:, fsel, :][...,[0,3]]!=0]
				rps = dps - pps
			else:
				rps = np.zeros(dps.shape, dtype=dps.dtype)
				LOGGER.debug(f"rps shape is {rps.shape}, flux shape is {flux.shape}")
				rps[:,1:,:] = dps[:,1:,:] - dps[:,:-1,:]
				rps[:,0,:] = rps[:,1,:] 
 
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", category=RuntimeWarning)
				if not allargs.data_rms:
					tmp = rps.compute()
				else:
					tmp = rps

				tmp[tmp==0] = np.nan

				if not allargs.data_rms:
					# rmss[loc, aa] = np.nanstd(tmp)*np.sqrt(2)	#np.sqrt(np.sum(tmp)/tmp.size) #
					chan_rms[:, aa] =  np.nanstd(tmp, axis=(0,2))*np.sqrt(2)
				else:
					chan_rms[:, aa] =  np.nanstd(tmp, axis=(0,2))

			# nv_nt[loc, aa] = __get_interval(rmss[loc, aa], flux[loc], n_ants[index[0], index[1], aa], SNR=SNR
			if aa%6 == 0:
				LOGGER.debug("Done computing noise for {%d}/{%d} antennas"%(aa+1, NUMBER_ANTENNAS))

		nv_nt[loc], grms_na[loc] = get_interval(chan_rms, flux[loc], n_ants[loc], SNR=SNR)


	LOGGER.debug("Took {} seconds to compute intervals".format(time.time()-time_c))

	# save products

	nv_nt[nv_nt==0] = np.nan

	nvis = np.nanmax(nv_nt)
	chunks_size_ok = True

	if nvis < fchunk:
		f_int = nvis
		t_int = 1
	else:
		f_int = fchunk
		t_int = np.ceil(nvis/fchunk)
		if t_int > tchunk:
			chunks_size_ok = False 

	LOGGER.info("Number visibilities per solution block is {}.".format(nvis))
	if chunks_size_ok:
		LOGGER.info("Suggested solution intervals based on chunks sizes frequency interval = {} and time interval = {}.".format(f_int, t_int))
	else:
		LOGGER.info("Suggested solution intervals frequency interval = {} and time interval = {} large than chunk sizes. Consider increasing the chunk sizes and reruning for a better suggestion.".format(f_int, t_int))
	
	if save_out:
		LOGGER.info("Computed statstics will be saved in the output folder.")
		
		np.save(outdir+"/"+_prefix+"num_antennas.npy", n_ants)
		np.save(outdir+"/"+_prefix+"chan_rms.npy", chan_rms)
		np.save(outdir+"/"+_prefix+"nv_nt.npy", nv_nt)
		np.save(outdir+"/"+_prefix+"grms.npy", grms_na)

		if datachunk is None:
			np.save(outdir+"/"+_prefix+"flags.npy", flags_ratio)

		imshow_stat(n_ants, outdir+"/"+_prefix+"nants.pdf")
		imshow_stat(chan_rms, outdir+"/"+_prefix+"chan_rms.pdf")
		imshow_stat(nv_nt, outdir+"/"+_prefix+"nv_nt.pdf")

		with np.errstate(divide='ignore', invalid='ignore'):
			imshow_stat(grms_na/128., outdir+"/"+_prefix+"grms-128.pdf")
			imshow_stat(grms_na/nv_nt, outdir+"/"+_prefix+"grms-var.pdf")



	LOGGER.info("Took {} seconds to complete".format(time.time()-t0))



def create_output_dirs(name, outdir):
	"""create ouput directory for pybdsm log files and output images"""

	if "/" in name:
		LOGGER.info("Output directory part of out-name will overwrite outdir option")
		outdir = os.path.dirname(name)

	if not outdir.endswith("/"):
		if outdir.endswith(".pc"):
			outdir += "/"
		else:
			outdir += ".pc/"

	if os.path.isdir(outdir):
		LOGGER.info("Output directory already exit from previous run, will make a backup")
		import glob
		nb_runs = len(glob.glob(outdir[:-1]+"*"))
		
		N = 0
		if nb_runs:
			backup_dir = outdir[:-1]+"-"+str(N)

		while os.path.exists(backup_dir):
			N += 1
			backup_dir = outdir[:-1]+"-"+str(N)

		import shutil
		shutil.move(outdir, backup_dir)

	os.mkdir(outdir)

	return outdir


def create_parser():
	p = argparse.ArgumentParser()
	p.add_argument("--ms", type=str, required=True, help="input measurement set (MS)")
	p.add_argument("--datacol", default="DATA", type=str, help="MS column containing the DATA to be calibrated")
	p.add_argument("--modelcol", default="MODEL_DATA", type=str, help="MS column containing the model visibilities (2GC only), can also be a tigger skymodel with a de tag (eg model.lsm.html@dE)")
	p.add_argument("--fluxcol", default="MODEL_DATA", type=str, help="MS column containing the model visibilities for the specific direction (3GC). Can also take difference of columns as in CubiCal")
	p.add_argument("--weightcol", default="WEIGHT", type=str, help="Weight Column")
	p.add_argument("--snr", default=3, type=int, help="minimum SNR of the solutions")
	p.add_argument("--min-bl", default=100, type=float, dest='minbl', help="exclude baselines less than set value")
	p.add_argument("--freq-chunk", default=128, type=int, dest='fchunk', help="size of frequency chunk to be use by CubiCal")
	p.add_argument("--time-chunk", default=64, type=int, dest='tchunk', help="size of time chunk to be use by CubiCal")
	p.add_argument("--single-chunk", default=None, type=str, dest='datachunk', help="use a specific datachunk like in CubiCal, example DOT0F1")
	p.add_argument('--cubical-flags', dest='cubi_flags', action='store_true', help="apply cubical flags otherwise only legacy flags are applied")

	p.add_argument("--rowchunks", default=4000, type=int, help="row chunks to be use by dask-ms")
	p.add_argument("--ncpu", default=0, type=int, help="number of CPUs to set dask multiprocessing")

	p.add_argument("--peakflux", default=None, type=float, help="peak flux in the skymodel if model visibilities are not yet computed")
	p.add_argument("--rms", default=None, type=float, help="rms to use if model visbilities are not yet computed")
	p.add_argument("--data-rms", default=False, action='store_true', help="use visibilities only to compute the rms")
	
	p.add_argument("--same", dest='same', action='store_true', help="use the same solution interval for time and frequency, default is use longer frequency interval")
	p.add_argument("--gaintable", type=str, help="gain table for second round search with Akaike Information Criterion (AIC)")
	p.add_argument("--gain-name", type=str, help="gain label to index CubiCal parameters database", default="G:gain", dest="Gname")
	p.add_argument('--usegains', dest='usegains', action='store_true', help="search using gains AIC")
	p.add_argument('--no-usegains', dest='usegains', action='store_false', help="do not search using gains AIC")
	p.add_argument('--usejhj', dest='usejhj', action='store_true', help="use the gains errors when computing the AIC")
	p.add_argument('--addjhj', dest='addjhj', action='store_true', help="add the noise")
	p.add_argument('--do2d', dest='do2d', action='store_true', help="do a 2D search")

	p.add_argument('--time-int', dest="tint", type=int, help="time interval use for the passed gains")
	p.add_argument('--freq-int', dest="fint", type=int, help="frequency interval use for the passed gains")


	p.add_argument("--outdir", type=str, default="out", help="output directory, default is created in current working directory")
	p.add_argument("--name", type=str, default="G", help="prefix to use in namimg output files")
	p.add_argument('--save-out', dest='save_out', action='store_true', help="save all computed statstics in npy files")

	p.add_argument("-v", "--verbose", help="increase output verbosity",
	                action="store_true")
	return p



def main():
	"""Main function."""
	LOGGER.info("Welcome to CubiInts")

	parser = create_parser()
	args = parser.parse_args()

	if args.verbose:
		for handler in LOGGER.handlers:
			handler.setLevel(logging.DEBUG)
	else:
		for handler in LOGGER.handlers:
			handler.setLevel(logging.INFO)

	LOGGER.info("started cubiints " + " ".join(sys.argv[1:]))


	if args.usegains is False:

		outdir = create_output_dirs(args.name, args.outdir)


		ms_opts = {"DataCol": args.datacol, "ModelCol": args.modelcol, "FluxCol": args.fluxcol, "WeightCol":args.weightcol, "msname": args.ms}

		if args.ncpu:
			ncpu = args.ncpu
			from multiprocessing.pool import ThreadPool
			dask.config.set(pool=ThreadPool(ncpu))
		else:
			import multiprocessing
			ncpu = multiprocessing.cpu_count()

		LOGGER.info("Using %i threads" % ncpu)

		try:
			compute_interval_dask_index(ms_opts=ms_opts, SNR=args.snr, dvis=False, outdir=outdir, figname=os.path.basename(args.name)+"-interval", row_chunks=args.rowchunks, minbl=args.minbl, 
										tchunk=args.tchunk, fchunk=args.fchunk, save_out=args.save_out, cubi_flags=args.cubi_flags, datachunk=args.datachunk, allargs=args)
		except:
			extype, value, tb = sys.exc_info()
			traceback.print_exc()
			pdb.post_mortem(tb)

		os.system('mv log-interval.txt %s/'%outdir)

	else:

		if args.tint is None or args.fint is None:
			print("options time-int and freq-int must be passed when usegains is selected")
			parser.exit()
		if args.gaintable is None:
			print("A gaintable must be specified")
			parser.exit()

		if args.do2d:
			tint = optimal_time_freq_interval_2D_same_gains(args.gaintable, args.ms, args.Gname, args.tint, args.fint, args.tchunk, verbosity=args.verbose, 																							prefix=os.path.basename(args.name), usejhj=(args.usejhj, args.addjhj)) # 2D -> from
		else:
			tint = optimal_time_freq_interval_from_gains(args.gaintable, args.ms, args.Gname, args.tint, args.fint, args.tchunk, verbosity=args.verbose, 
																							prefix=os.path.basename(args.name), usejhj=(args.usejhj, args.addjhj)) # 2D -> from
		print("optimal interval time-int= {}".format(tint))
		print("use jhj was ", args.usejhj) 
		
