#!/usr/bin/env python

# trying to daskify this code properly using surfchi2 from surfvis
import argparse
import matplotlib as mpl
mpl.rcParams.update({'font.size': 11, 'font.family': 'serif'})
import os
import sys
import numpy as np
import xarray as xr
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from daskms import xds_from_ms, xds_from_table
from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
import random

import sys
import os

import traceback

try:
	import ipdb as pdb
except:
	import pdb

from cubiints.dtools import rms_chan_ant, makeplot, model_from_lsm

import time as tt
import logging 
mpl_logger = logging.getLogger('matplotlib') 
mpl_logger.setLevel(logging.WARNING)

from cubiints import LOGGER

def compute_interval_dask_index(ms_opts={}, outdir="./soln-intervals", field_id=0,
                                tchunk=64, fchunk=128, use_corrs=[0,-1], nthreads=4, doplots=True, maxgroups=12, snr=3, name="G"):
    
    msname = ms_opts["msname"]
    
    # chunking info 
    schema = {}
    schema['FLAG'] = {'dims': ('chan', 'corr')}
    xds = xds_from_ms(msname,
					  chunks={'row': -1},
					  columns=['TIME', 'FLAG'],
					  group_cols=['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'])
    
    chunks = []
    rbin_idx = []
    rbin_counts = []
    tbin_idx = []
    tbin_counts = []
    fbin_idx = []
    fbin_counts = []
    t0s = []
    tfs = []
    n_scans = len(xds)
    for ds in xds:
        time = ds.TIME.values
        ut, counts = np.unique(time, return_counts=True)
        if tchunk in [0, -1]:
            utpc = ut.size
        else:
            utpc = tchunk
        row_chunks = [np.sum(counts[i:i+utpc])
                 	  for i in range(0, ut.size, utpc)]

        nchan = ds.chan.size
        if fchunk in [0, -1]:
            fchunk = nchan

        # list per ds
        chunks.append({'row': tuple(row_chunks), 'chan': fchunk})

        ridx = np.zeros(len(row_chunks))
        ridx[1:] = np.cumsum(row_chunks)[0:-1]
        rbin_idx.append(da.from_array(ridx.astype(int), chunks=1))
        rbin_counts.append(da.from_array(row_chunks, chunks=1))

        ntime = ut.size
        tidx = np.arange(0, ntime, utpc)
        tbin_idx.append(da.from_array(tidx.astype(int), chunks=1))
        tidx2 = np.append(tidx, ntime)
        tcounts = tidx2[1:] - tidx2[0:-1]
        tbin_counts.append(da.from_array(tcounts, chunks=1))

        t0 = ut[tidx]
        t0s.append(da.from_array(t0, chunks=1))
        tf = ut[tidx + tcounts -1]
        tfs.append(da.from_array(tf, chunks=1))

        fidx = np.arange(0, nchan, fchunk)
        fbin_idx.append(da.from_array(fidx, chunks=1))
        fidx2 = np.append(fidx, nchan)
        fcounts = fidx2[1:] - fidx2[0:-1]
        fbin_counts.append(da.from_array(fcounts, chunks=1))
    
    _model_col = False
    col2 = False

    if "lsm.html" in ms_opts["ModelCol"]:
        LOGGER.info("Tigger skymodel pass as model, will compute the noise from the data column only")
        columns = ["ANTENNA1", "ANTENNA2", "TIME", "FLAG", "FLAG_ROW", ms_opts["DataCol"], ms_opts["WeightCol"], "MODEL_DATA"]  #,
    else:
        columns = ["ANTENNA1", "ANTENNA2", "TIME", "FLAG", "FLAG_ROW", ms_opts["DataCol"], ms_opts["ModelCol"], ms_opts["WeightCol"]]
        if ms_opts["FluxCol"]:
            col2 = ms_opts["FluxCol"].split("-")[1]
            columns.append(col2)
        _model_col = True
        
    
    datacol =  ms_opts["DataCol"]
    weightcol = ms_opts["WeightCol"]
    modelcol = "MODEL_DATA" if "lsm.html" in ms_opts["ModelCol"] else ms_opts["ModelCol"]
    
    table_schema = {datacol: {'dims': ('chan', 'corr')}, modelcol: {'dims': ('chan', 'corr')}, 'FLAG':{'dims': ('chan', 'corr')}}
    if col2:
        table_schema[col2] = {'dims':('chan', 'corr')}
    
    if ms_opts["WeightCol"] == "WEIGHT":
        table_schema[ms_opts["WeightCol"]] = {'dims':('corr',)}
    else:
        table_schema[ms_opts["WeightCol"]] = {'dims':('chan', 'corr')}
    
    LOGGER.info(f"Found {n_scans} groups, {rbin_counts[0].shape[0]} time chunks and {fbin_counts[0].shape[0]} frequency chunks.")

    xds = xds_from_ms(msname,
                    columns=columns,
                    chunks=chunks,
                    group_cols=['FIELD_ID', 'DATA_DESC_ID', 'SCAN_NUMBER'],
                    table_schema = table_schema
                    )
    
    xds = [xd for xd in xds if xd.FIELD_ID == field_id]
    
    if n_scans>maxgroups:
        xds = random.sample(xds, maxgroups)
        LOGGER.info(f"{maxgroups} out of the {n_scans} groups will be used for the search!")

    out_ds = []
    intervals = np.zeros(len(xds))
    idts = []

    for i, ds in enumerate(xds):
        ds = ds.sel(corr=use_corrs)

        data = ds.get(datacol).data
        
        weight = ds.get(weightcol).data
        if weight.ndim != 3:
            weight = da.broadcast_to(weight[:,None,:], data.shape)
		
        flag = ds.get("FLAG").data
        flag_row = ds.get("FLAG_ROW").data
        flag = da.logical_or(flag, flag_row[:,np.newaxis,np.newaxis])

        weight *= da.logical_not(flag)
        ant1 = ds.ANTENNA1.data
        ant2 = ds.ANTENNA2.data

        # apply weights:
        data *= weight
        if _model_col:
            model = ds.get(modelcol).data
            if col2:
                modelsub = ds.get(col2).data
                model -= modelsub
            model *=weight
        else:
            complexflux = model_from_lsm(ms_opts["ModelCol"])
            model = ds.get("MODEL_DATA").data
            model[:,:,:] = complexflux
            model *=weight

        field = ds.FIELD_ID
        ddid = ds.DATA_DESC_ID
        scan = ds.SCAN_NUMBER

        tmp_rms = rms_chan_ant(data, model, flag, ant1, ant2,
				    rbin_idx[i], fbin_idx[i], fbin_counts[i], tbin_counts[i], fchunk, snr)
        
        d = xr.Dataset(
			data_vars={'data': (('time', 'freq', 'nfreq', 'antenna', 'corr'), tmp_rms),
					   'fbin_idx': (('freq'), fbin_idx[i]),
					   'fbin_counts': (('freq'), fbin_counts[i]),
					   't0s': (('time'), t0s[i]), #t0s
					   'tfs': (('time'), tfs[i]), #tfs
                       },
			attrs = {'FIELD_ID': ds.FIELD_ID,
					 'DATA_DESC_ID': ds.DATA_DESC_ID,
					 'SCAN_NUMBER': ds.SCAN_NUMBER,
                    },
		)

        idt = f'::F{ds.FIELD_ID}_D{ds.DATA_DESC_ID}_S{ds.SCAN_NUMBER}'
        out_ds.append(xds_to_zarr(d, outdir+"/" + idt))
        idts.append(idt)

    with ProgressBar():
        dask.compute(out_ds, optimize_graph=True, num_workers=nthreads) #  

    #--------plot the results---------------------#
    #'nvis':(('nvis'), nvis)

    for j, idt in enumerate(idts):
        xds = xds_from_zarr(outdir+"/" + idt)
        for ds in xds:
            field = ds.FIELD_ID
            if not os.path.isdir(outdir+"/" + f'/field{field}'):
                os.system('mkdir '+ outdir+"/"+ f'/field{field}')

            spw = ds.DATA_DESC_ID    
            scan = ds.SCAN_NUMBER
            
        tmp = ds.data.values
        t0s = ds.t0s.values
        tfs = ds.tfs.values
        fbin_idx = ds.fbin_idx.values
        fbin_counts = ds.fbin_counts.values
        intervals[j] = np.max(tmp[...,3])
        t, f,_,_ = np.unravel_index(np.nanargmax(tmp[...,3]), tmp[...,3].shape)

        if doplots:
            basename = outdir + f'/field{field}/SPW{spw}-SCAN{scan}-'
        
            makeplot(tmp[t,f,:,:,3], basename + f'T{t}F{f}-nv_nt.png',
                    t0s[t], tf[t], fbin_idx[f], fbin_idx[f] + fbin_counts[f])

            makeplot(tmp[t,f,:,:,0], basename + f'T{t}F{f}-rms.png',
                    t0s[t], tf[t], fbin_idx[f], fbin_idx[f] + fbin_counts[f])

            makeplot(tmp[t,f,:,:,1], basename + f'T{t}F{f}-ant.png',
                    t0s[t], tf[t], fbin_idx[f], fbin_idx[f] + fbin_counts[f])
            
            makeplot(tmp[t,f,:,:,2], basename + f'T{t}F{f}-modelabs.png',
                    t0s[t], tf[t], fbin_idx[f], fbin_idx[f] + fbin_counts[f])
    
        LOGGER.info("Scan {}/{} done!".format(j+1, len(intervals)))
        
    # Aggregate the results ------------#
    LOGGER.info("Output intervals {}.".format(intervals))
    nvis = np.max(intervals)
    chunks_size_ok = True
    f_int = np.ceil(np.sqrt(nvis))
    # if nvis < fchunk:
    #     f_int = nvis
    #     t_int = 1
    # else:
    #     f_int = fchunk
    t_int = np.ceil(nvis/f_int)
    if t_int > tchunk or f_int>fchunk:
        chunks_size_ok = False 

    LOGGER.info("Number visibilities per solution block is {}.".format(nvis))
    if chunks_size_ok:
        LOGGER.info("Suggested solution intervals based on chunks sizes frequency interval = {} and time interval = {}.".format(f_int, t_int))
    else:
        LOGGER.info("Suggested solution intervals frequency interval = {} and time interval = {} large than chunk sizes. Consider increasing the chunk sizes and reruning for a better suggestion.".format(f_int, t_int))

    file = open(outdir+f"/{name}-solution-intervals.txt", "w")
    file.write(f"{name}.time_interval = {int(t_int):d}\n")
    file.write(f"{name}.freq_interval = {int(f_int):d}")
    file.close()

    
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
    p.add_argument("--fluxcol", default=None, type=str, help="MS column containing the model visibilities for the specific direction (3GC). Can also take difference of columns as in CubiCal (D1-D2)")
    p.add_argument("--weightcol", default="WEIGHT", type=str, help="Weight Column")
    p.add_argument("--snr", default=3, type=int, help="minimum SNR of the solutions")
    p.add_argument("--freq-chunk", default=128, type=int, dest='fchunk', help="size of frequency chunk to be use by CubiCal, avoid chunks bigger then 128")
    p.add_argument("--time-chunk", default=64, type=int, dest='tchunk', help="size of time chunk to be use by CubiCal")

    p.add_argument("--nthreads", default=12, type=int, help="number of dask threads to use")
    p.add_argument("--max-scans", default=12, type=int, help="maximum of number of groups (scans and spws) to use for the search, for very large datasets")
    p.add_argument("--field-id", default=0, type=int, help="which field to use")

    p.add_argument("--outdir", type=str, default="out", help="output directory, default is created in current working directory")
    p.add_argument("--name", type=str, default="G", help="prefix to use in namimg output files (should be the gain term for quartical")
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

    LOGGER.info("started snr-ints " + " ".join(sys.argv[1:]))

    outdir = create_output_dirs(args.name, args.outdir)

    ms_opts = {"DataCol": args.datacol, "ModelCol": args.modelcol, "FluxCol": args.fluxcol, "WeightCol":args.weightcol, "msname": args.ms}

    try:
        t0 = tt.time()
        compute_interval_dask_index(ms_opts=ms_opts, outdir=outdir, tchunk=args.tchunk, fchunk=args.fchunk, field_id=args.field_id,
                                            use_corrs=[0,-1], nthreads=args.nthreads, doplots=args.save_out, maxgroups=args.max_scans, snr=args.snr, name=args.name)

        LOGGER.info(f"Completed in {(tt.time()-t0)/60:.2f} mins")
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

    os.system('mv log-interval.txt %s/'%outdir)
    
