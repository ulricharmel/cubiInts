import argparse
from pyrap.tables import table
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import os

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


xds = []

LOGGER = create_logger(__name__)


def model_flux_per_scan_dask(bounds, scan_ids, fluxcols, w, f, filename="M1", outdir="./soln-intervals"):
	"""compute the flux per interval scans"""

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

	LOGGER.info("Done applying weights and flags")

	flux = np.zeros(len(scan_ids))

	for i, scan_id in enumerate(scan_ids):
		print("computing Flux for scan %d"%i)

		if bounds[i+1] == -1:
			sel = slice(bounds[i], None)
		else:
			sel = slice(bounds[i], bounds[i+1])
		
		if __sub_model:
			model_abs = da.absolute(m1[sel][...,[0,3]][m0[sel][...,[0,3]]!=0] - m0[sel][...,[0,3]][m0[sel][...,[0,3]]!=0])
		else:
			model_abs = da.absolute(m0[sel][...,[0,3]][m0[sel][...,[0,3]]!=0])
		
		flux[i] = np.mean(model_abs.compute())


	np.save(outdir+"/"+filename+"model.npy", flux)

	return flux


def compute_interval_dask(ms_opts={}, SNR=3, dvis=False, outdir="./soln-intervals", figname="interval", row_chunks=4000, freqslice=slice(None), jump=1, minbl=100):
	"""replicate the compute interval using dask arrays"""

	def __get_interval(rms, P, Na, SNR=3):
		Nvis = int(np.ceil(SNR**2.*rms**2./((int(Na)-1.)*P**2.)))
		return Nvis

	t0 = time.time()

	t = table(ms_opts["msname"])

	f = build_flag_colunm(t, minbl=minbl, obvis=None, freqslice=freqslice, row_chunks=row_chunks)
	LOGGER.info("finished building flags")

	w = fetch(ms_opts["WeightCol"], subset=t, return_dask=True, row_chunks=row_chunks)
	LOGGER.info("read weight-column susscessful")

	LOGGER.info("Table Closed")

	cols = ms_opts["FluxCol"].split("-")

	columns = list(set(["ANTENNA1", "ANTENNA2", "TIME", "SCAN_NUMBER", ms_opts["DataCol"], ms_opts["ModelCol"]]) | set(cols))

	LOGGER.info("Reading the columns: [{}] as a daskms".format(", ".join('{}'.format(col) for col in columns)))

	global xds

	xds = xds_from_ms(ms_opts["msname"], columns=columns, chunks={"row":row_chunks})

	scans = getattr(xds[0], "SCAN_NUMBER").data.compute()
	time_col = getattr(xds[0], "TIME").data.compute()
	ant1 = getattr(xds[0], "ANTENNA1").data.compute()
	ant2 = getattr(xds[0], "ANTENNA2").data.compute()
	n_ants = get_mean_n_ant_scan(ant1, ant2, f.compute(), scans, time_col, jump)

	scan_ids, bounds =  get_scan_bounds(scans, jump)

	LOGGER.info("number scans is {}".format(len(scan_ids)))

	if dvis:
		raise NotImplementedError("Compute rms from visibilities only yet to be implemented")


	d = getattr(xds[0], ms_opts["DataCol"]).data
	p = getattr(xds[0], ms_opts["ModelCol"]).data

	#apply flags
	d*=(f==False)
	# p*=(f==False) select based on d only

	#apply weights
	d*=w
	p*=w

	LOGGER.info("Done applying weights and flags")

	_prefix = figname.split("interval")[0]

	flux = model_flux_per_scan_dask(bounds, scan_ids, cols, w, f, filename=_prefix, outdir=outdir)
	rmss = np.zeros(len(scan_ids))

	Na_ms = max(ant2) + 1

	nv_nt = np.zeros(len(scan_ids))
	nv_nt2 = np.zeros(len(scan_ids))

	# import pdb; pdb.set_trace()

	for i, scan_id in enumerate(scan_ids):
		print("computing interval for scan %d"%i)

		if bounds[i+1] == -1:
			sel = slice(bounds[i], None)
		else:
			sel = slice(bounds[i], bounds[i+1])

		if dvis:
			raise NotImplementedError("Compute rms from visibilities only yet to be implemented")
		else:
			dps = d[sel][...,[0,3]][d[sel][...,[0,3]]!=0]
			pps = p[sel][...,[0,3]][d[sel][...,[0,3]]!=0]
			rps = dps - pps

		rmss[i] = np.std(rps.compute())  # *np.sqrt(2)	
		
		nv_nt[i] = __get_interval(rmss[i], flux[i], n_ants[i], SNR=SNR)
		nv_nt2[i] = __get_interval(rmss[i], flux[i], Na_ms, SNR=SNR)


	LOGGER.info("Number of antennas per scan: [{}]".format(", ".join('{:.2f}'.format(x) for x in n_ants)))
	LOGGER.info("Esitmated rms per scan: [{}]".format(", ".join('{:.2f}'.format(x) for x in rmss)))
	LOGGER.info("Esitmated flux per scan: [{}]".format(", ".join('{:.2f}'.format(x) for x in flux)))
	LOGGER.info("Soln-interval per scan: [{}]".format(", ".join('{:.0f}'.format(x) for x in nv_nt)))

	# save products
	
	np.save(outdir+"/"+_prefix+"rms.npy", rmss)
	np.save(outdir+"/"+_prefix+"num_antennas.npy", n_ants)


	if freqslice == slice(None):
		slicestr = "all"
	else:
		slicestr = "%d-%d"%(freqslice.start, freqslice.stop)

	fig, ax1 = plt.subplots()
	ax1.plot(scan_ids, nv_nt, c='b', linewidth="3", marker="D", label="mean Na")
	ax1.plot(scan_ids, nv_nt2, c='r', linewidth="3", marker="D", label="fixed Na")

	# zip joins x and y coordinates in pairs
	for x,y in zip(scan_ids, nv_nt):
		label = "{:.0f}".format(y)
		ax1.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

	for x,y in zip(scan_ids, nv_nt2):
		label = "{:.0f}".format(y)
		ax1.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

	ax1.set_ylabel("Soln-interval", size=30)
	ax1.set_xlabel("Scan index", size=30)
	ax1.set_title(figname, fontdict={'fontsize': 8, 'fontweight': 'medium'})
	#plt.legend(loc='best', fontsize='x-small')
	fig.tight_layout()
	fig.savefig(outdir+"/"+figname+"_jump_scan_%s.pdf"%slicestr, dpi=200)
	plt.clf()
	plt.close()

	LOGGER.info("Took {} seconds to complete".format(time.time()-t0))


def create_output_dirs(outdir):
	"""create ouput directory for pybdsm log files and output images"""

	outdir = outdir+".pc"
	if os.path.isdir(outdir):
		LOGGER.info("Output directory already exit from previous run, will make a backup")
		import glob
		nb_runs = len(glob.glob(outdir+"*"))
		
		import shutil
		shutil.move(outdir, outdir+"-"+str(nb_runs-1))

	os.mkdir(outdir)

	return outdir


def create_parser():
	p = argparse.ArgumentParser()
	p.add_argument("--ms", type=str, required=True, help="input measurement set (MS)")
	p.add_argument("--datacol", default="DATA", type=str, help="MS column containing the DATA to be calibrated")
	p.add_argument("--modelcol", default="MODEL_DATA", type=str, help="MS column containing the model visibilities (2GC only)")
	p.add_argument("--fluxcol", default="MODEL_DATA", type=str, help="MS column containing the model visibilities for the specific direction (3GC). Can also take difference of columns as in CubiCal")
	p.add_argument("--weightcol", default="WEIGHT", type=str, help="Weight Column")
	p.add_argument("--snr", default=3, type=int, help="minimum SNR of the solutions")
	p.add_argument("--min-bl", default=100, type=float, dest='minbl', help="exclude baselines less than set value")

	p.add_argument("--rowchunks", default=4000, type=int, help="row chunks to be use by dask-ms")
	p.add_argument("--ncpu", default=0, type=int, help="number of CPUs to set dask multiprocessing")

	p.add_argument("--peakflux", default=None, type=float, help="peak flux in the skymodel if model visibilities are not yet computed")
	p.add_argument("--rms", default=None, type=float, help="rms to use if model visbilities are not yet computed")
	
	p.add_argument("--same", dest='same', action='store_true', help="use the same solution interval for time and frequency, default is use longer frequency interval")
	p.add_argument("--gaintable", type=str, help="gain table for second round search with Akaike Information Criterion (AIC)")
	p.add_argument('--usegains', dest='usegains', action='store_true', help="search using gains AIC")
	p.add_argument('--no-usegains', dest='usegains', action='store_false', help="do not search using gains AIC")
	p.add_argument('--time-int', dest="tint", type=int, help="time interval use for the passed gains")
	p.add_argument('--freq-int', dest="fint", type=int, help="frequency interval use for the passed gains")

	p.add_argument("--outdir", type=str, default="out", help="output directory, default is created in current working directory")
	p.add_argument("--name", type=str, default="G", help="prefix to use in namimg output files")

	p.set_defaults(usegains=False)

	p.add_argument("-v", "--verbose", help="increase output verbosity",
	                action="store_false")
	return p



def main():
	"""Main function."""
	LOGGER.info("Welcome to CubiInts")

	parser = create_parser()
	args = parser.parse_args()

	outdir = create_output_dirs(args.outdir)


	ms_opts = {"DataCol": args.datacol, "ModelCol": args.modelcol, "FluxCol": args.fluxcol, "WeightCol":args.weightcol, "msname": args.ms}

	if args.ncpu:
		ncpu = args.ncpu
		from multiprocessing.pool import ThreadPool
		dask.config.set(pool=ThreadPool(ncpu))
	else:
		import multiprocessing
		ncpu = multiprocessing.cpu_count()

	LOGGER.info("Using %i threads" % ncpu)


	compute_interval_dask(ms_opts=ms_opts, SNR=args.snr, dvis=False, outdir=outdir, figname=args.name+"-interval", row_chunks=args.rowchunks, minbl=args.minbl)

