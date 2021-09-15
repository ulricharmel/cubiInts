import os
import numpy as np
import dask.array as da
from numba import njit
import Tigger
import warnings
import line_profiler
profile = line_profiler.LineProfiler()

# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 11, 'font.family': 'serif'})
# mpl.use('Agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings("ignore", category=RuntimeWarning)

from cubiints import LOGGER

SNR = 3

def __get_interval(rms, P, Na):

    # import ipdb; ipdb.set_trace()
    
    if np.isnan(P) or np.isnan(rms) or np.isnan(Na) or Na<2:
        return 0, np.nan
    else:
        Nvis = int(np.ceil(SNR**2.*rms**2./((Na-1.)*P**2.)))
        grms = rms**2./((Na-1.)*P**2.) 
        return Nvis, grms

get_interval = np.vectorize(__get_interval)

def rms_chan_ant(data, model, flag, ant1, ant2,
           rbin_idx, rbin_counts, fbin_idx, fbin_counts, tbin_counts):

	nant = da.maximum(ant1.max(), ant2.max()).compute() + 1
	nch = fbin_counts[0].compute()
	res = da.blockwise(_rms_chan_ant, 'tfnp5',
                       data, 'tfc',
                       model, 'tfc',
                       flag, 'tfc',
                       ant1, 't',
                       ant2, 't',
                       rbin_idx, 't',
                       rbin_counts, 't',
                       fbin_idx, 'f',
                       fbin_counts, 'f',
					   tbin_counts, 't',
                       align_arrays=False,
                       dtype=np.float64,
                       adjust_chunks={'t': rbin_idx.chunks[0],
                                      'f': fbin_idx.chunks[0],},
                       new_axes={'n':nch, 'p': nant, '5': 5})
	return res

# @njit(fastmath=True, nogil=True)
def _rms_chan_ant(data, model, flag, ant1, ant2,
           rbin_idx, rbin_counts, fbin_idx, fbin_counts, tbin_counts):
	
	nrow, nchan, ncorr = data[0].shape

	nto = rbin_idx.size
	nfo = fbin_idx.size
	uant1 = np.unique(ant1)
	uant2 = np.unique(ant2)
	nant = np.maximum(uant1.max(), uant2.max()) + 1
	nch = fbin_counts[0]

	# account for chunk indexing
	rbin_idx2 = rbin_idx - rbin_idx.min()
	fbin_idx2 = fbin_idx - fbin_idx.min()
	
	# init output array
	out = np.zeros((nto, nfo, nch, nant, 5), dtype=np.float64)

	# import pdb; pdb.set_trace()

	# nch = fbin_counts[0]
	# ant1p = 1+np.repeat(ant1p[:,np.newaxis], nch, axis=1)
	# ant2p = 1+np.repeat(ant2p[:,np.newaxis], nch, axis=1)

	for t in range(nto):
		rowi = rbin_idx2[t]
		rowf = rbin_idx2[t] + rbin_counts[t]
		datar = data[0][rowi:rowf]
		flagr = flag[0][rowi:rowf].astype(float)
		ant1p = ant1[rowi:rowf]
		ant2p = ant2[rowi:rowf]

		modelr = model[0][rowi:rowf]
		
		for f in range(nfo):
			chani = fbin_idx2[f]
			chanf = fbin_idx2[f] + fbin_counts[f]
			data2 = datar[:,chani:chanf]
			flag2 = flagr[:,chani:chanf]
			model2 = modelr[:,chani:chanf]


			with warnings.catch_warnings():
				warnings.simplefilter("ignore", category=RuntimeWarning)
				for aa in range(nant):
					dps = data2[(ant1p==aa)|(ant2p==aa)]
					fps = flag2[(ant1p==aa)|(ant2p==aa)]
					# dps.compute_chunk_sizes()
					# rps = np.zeros(dps.shape, dtype=dps.dtype)
					rps = dps[:,1:,:] - dps[:,:-1,:]
					# rps[:,0,:] = rps[:,1,:]
					
					rps[rps==0] = np.nan
					fps[fps!=0] = np.nan
					fps[fps==0] = 1.

					out[t,f, 1:, aa, 0] =  np.nanstd(rps, axis=(0,2))
					out[t,f, 0, aa, 0] =  out[t,f, 1, aa, 0]
					out[t,f, :, aa, 1] =  (np.nansum(fps[...,0], axis=0))/(tbin_counts[t])
					# compute the model in brute force way
					# if model_col:
					mps = model2[(ant1p==aa)|(ant2p==aa)]
					mps = da.absolute(mps)
					mps[mps==0] = np.nan
					out[t,f, :, aa, 2] = np.nanmean(mps, axis=(0,2)) 

					out[t,f, :, aa, 3], out[t, f, :, aa, 4]  = get_interval(out[t,f,:,aa,0], out[t,f,:,aa,2], out[t,f,:,aa,1])
					# else:
						# out[:, aa, 2] = flux
	
	return out

def model_from_lsm(lsmmodel):
    """Return a flux to use form an lsmodel
    Args:
        lsmmodel (str)
        -Tigger sky model
    Returns:
        complexflux (complex number)
        -flux + 0j
    """
   

    if "@" in lsmmodel:
        modelname, tag = lsmmodel.split("@")
        de = {}
        model_s= Tigger.load(modelname)
        for src in model_s.sources:
            if src.getTag(tag):
                de[src.cluster] = src.cluster_flux

        fluxes = da.array(list(de.values()))
        complexflux = np.min(fluxes) + 0j
    else:
        model_s = Tigger.load(lsmmodel)
        clusters = {}
        for src in model_s.sources:
            try:
                clusters[src.cluster] = src.cluster_flux
            except AttributeError:
                clusters[src.name] = src.flux.I

        fluxes = da.array(list(clusters.values()))
        complexflux = np.sqrt(np.sum(fluxes**2)) + 0j
    
    return complexflux


def makeplot(array, savename, t0, tf, chan0, chanf):
	"""do an imshow of the computed stats"""

	cmap = 'jet' #'cubehelix'  #'grayas'
	stretch = 'linear'
	array[array==0] = np.nan

	fig, ax1 = plt.subplots()
	img = plt.imshow(array.squeeze(), cmap=cmap, aspect="auto")
	cb = fig.colorbar(img, pad=0.01)
	# cb.set_label("Noise [Jy]",size=30)
	ax1.set_ylabel("Channel index", size=15)
	ax1.set_xlabel("Antenna index", size=15)
	plt.tick_params(labelsize=15)
	fig.tight_layout()
	fig.suptitle(f't0 = {t0}, tf = {tf}, chan {chan0}-{chanf}', fontsize=10)
	fig.savefig(savename, dpi=250)
	plt.close(fig)