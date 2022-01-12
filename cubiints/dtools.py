import os
import numpy as np
import dask.array as da
from numba import njit, prange, generated_jit, vectorize
from numba.typed import List
import Tigger
import warnings
import line_profiler
profile = line_profiler.LineProfiler()

# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 11, 'font.family': 'serif'})

warnings.filterwarnings("ignore", category=RuntimeWarning)

from cubiints import LOGGER

@njit(fastmath=True, nogil=True, parallel=True, cache=True)
def get_interval(rms, P, Na, SNR, Na_min):
	Nvis = np.empty(len(rms))
	grms = np.empty(len(rms))

	for i in prange(len(rms)):
		if np.isnan(P[i]): 
			rr, pp = 0, np.nan
		elif np.isnan(rms[i]):
			rr, pp = 0, np.nan
		elif np.isnan(Na[i]):
			rr, pp = 0, np.nan
		elif Na[i]<Na_min:
			rr, pp = 0, np.nan
		else:
			rr = int(np.ceil(SNR**2.*rms[i]**2./((Na[i]-1.)*P[i]**2.)))
			pp = rms[i]**2./((Na[i]-1.)*P[i]**2.) 

		Nvis[i], grms[i] = rr, pp
	
	return Nvis, grms

@njit(fastmath=True, nogil=True, parallel=True, cache=True)
def np_apply_along_2axis(func1d, arr):
    # axis (0 and 2, we assume the data ndim is 3)
    result = np.empty(arr.shape[1])
    for i in prange(len(result)):
        result[i] = func1d(arr[:, i, :])
    
    return result

@njit(fastmath=True, nogil=True, parallel=True, cache=True)
def np_apply_along_axis(func1d, arr):
    # axis (0 we assume the data ndim is 2)
    result = np.empty(arr.shape[1])
    for i in prange(len(result)):
        result[i] = func1d(arr[:, i])
    
    return result

@njit
def np2Dsum(array):
    return np_apply_along_2axis(np.nansum, array) #nanmean

@njit
def np2Dstd(array):
    return np_apply_along_2axis(np.nanstd, array)

@njit
def np1Dsum(array):
    return np_apply_along_axis(np.nansum, array)


def rms_chan_ant(data, model, flag, ant1, ant2,
           rbin_idx, fbin_idx, fbin_counts, tbin_counts, nch, snr):

	nant = da.maximum(ant1.max(), ant2.max()).compute() + 1
	# nch = fbin_counts[0].compute()
	res = da.blockwise(_rms_chan_ant, 'tfnp5',
                       data, 'tfc',
                       model, 'tfc',
                       flag, 'tfc',
                       ant1, 't',
                       ant2, 't',
                       fbin_counts, 'f',
					   tbin_counts, 't',
					   snr, None,
					   concatenate=True,
                       align_arrays=False,
                       dtype=np.float64,
                       adjust_chunks={'t': rbin_idx.chunks[0],
                                      'f': fbin_idx.chunks[0],},
                       new_axes={'n':nch, 'p': nant, '5': 5})
	return res

@njit(fastmath=True, nogil=True, parallel=True, cache=True)
def _rms_chan_ant(data, model, flag, ant1, ant2, fbin_counts, tbin_counts, snr):
	
	uant1 = np.unique(ant1)
	uant2 = np.unique(ant2)
	nant = np.maximum(uant1.max(), uant2.max()) + 1
	na_min = nant/3
	nch = fbin_counts[0]

	# init output array
	out = np.zeros((1,1, nch, nant, 5), dtype=np.float64)
		
	for aa in prange(nant):
		dps = data[(ant1==aa)|(ant2==aa)]
		fps = flag[(ant1==aa)|(ant2==aa)]
		
		rps = dps[:,1:,:] - dps[:,:-1,:]
	
		unflagged = np1Dsum(fps[...,0])
		
		out[0, 0, 1:, aa, 0] =  np.sqrt(np2Dsum(np.abs(rps)**2)) #/(2*2*unflagged_bls[1:])  # np2Dstd
		out[0, 0, 0, aa, 0] =  out[0, 0, 1, aa, 0]
		out[0, 0, :, aa, 0] /= np.sqrt(2*2*2*unflagged)
		
		out[0, 0, :, aa, 1] =  unflagged/tbin_counts[0]

		mps = model[(ant1==aa)|(ant2==aa)]
		mps = np.abs(mps)
		
		out[0, 0, :, aa, 2] = np2Dsum(mps)/(unflagged*2) 

		out[0, 0, :, aa, 3], out[0, 0, :, aa, 4]  = get_interval(out[0, 0, :,aa,0], out[0, 0, :,aa,2], out[0, 0, :,aa,1], snr, na_min)
			
	
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
   
	modelname, tag = lsmmodel.split("@")
	model_s = Tigger.load(modelname)
	clusters = {}
	for src in model_s.sources:
		try:
			clusters[src.cluster] = src.cluster_flux
		except AttributeError:
			clusters[src.name] = src.flux.I

	fluxes = da.array(list(clusters.values()))
	fullmodel = np.sqrt(np.sum(fluxes**2)) + 0j
	
	if "@" in lsmmodel:
		# modelname, tag = lsmmodel.split("@")
		de = {}
		# model_s= Tigger.load(modelname)
		for src in model_s.sources:
			if src.getTag(tag):
				de[src.cluster] = src.cluster_flux

		fluxes = da.array(list(de.values()))
		sourcemodel = np.min(fluxes) + 0j # min
	else:
		sourcemodel = fullmodel
		

	# LOGGER.info(f"flux is {fullmodel.real.compute()}, {sourcemodel.real.compute()}")
    
	return fullmodel, sourcemodel


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