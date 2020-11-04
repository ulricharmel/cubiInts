import numpy as np
import numba
from pyrap.tables import table

import sys
sys.path.insert(0, '/net/jake/home/ulrich/roibosenv/CubiCal')
# reload(sys)
# sys.setdefaultencoding('utf-8')

import cubical.param_db as db
import matplotlib.pyplot as plt
from matplotlib import colors

import matplotlib.mlab as mlab
from matplotlib import rcParams
rcParams.update({'font.size': 20, 'font.family': 'sans-serif'})
plt.rcParams["figure.figsize"] = [16,9]
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

# from cubiints.tools import create_logger
# LOGGER = create_logger("AIC")

from cubiints import LOGGER


def extract_from_db(dbfile, name="G:gain", return_freq=False):
	"""Extract from dbfile"""

	LOGGER.debug("Loading gains database {}".format(dbfile))

	try:
		dbdata = db.load(dbfile)
		gains = dbdata[name]

		gainscube = gains.get_cube()
		data = gainscube.data[-1] # remove n_dir

		freqs = gains.grid[gains.ax.freq] / 1.0e9

	except Exception as e:
		# print(e)
		data = np.load(dbfile)
		data = data[-1] #,0] # assuming ntchunks is 1, remove n_dir

		if return_freq:
			freqs = np.load(dbfile[:-4]+"_freqs.npy")

	if return_freq:
		return data, freqs
	else:
		return data

def compute_gains_chisq(gobs, avggains, tint=1, fint=1, gerr=1, cross_corr=False, mean_std=1, usejhj=False, fz=64, tz=1):
	"""compute the chisq of the gains"""

	n_tchunks, n_dir, n_tim, n_fre, n_ant, n_cor, n_cor = gobs.shape
	resid = gobs - avggains

	t_bins = range(0, n_tim, tint)
	f_bins = range(0, n_fre, fint)

	n_ef = len(range(0, n_fre, fz))
	n_et = len(range(0, n_tim, tz))

	n_eq_cor = max(n_fre - n_ef, 1)*max(n_tim - n_et, 1)
	n_val_cor = max(len(f_bins)-n_ef, 1)*max(len(t_bins)-n_et, 1) 

	if cross_corr is False:
		resid[...,(0,1), (1,0)] = 0
		num_valid_eqs = n_tchunks*n_dir*n_tim*n_fre*n_ant*n_cor
		n_valid_sols = n_tchunks*n_dir*len(t_bins)*len(f_bins)*n_ant*n_cor
	else:
		num_valid_eqs = n_tchunks*n_dir*n_tim*n_fre*n_ant*n_cor*n_cor
		n_valid_sols = n_tchunks*n_dir*len(t_bins)*len(f_bins)*n_ant*n_cor*n_cor

	gerr[gerr==0] = 1 # To avoid division by zero

	# print(gerr.shape, resid.shape)

	if usejhj:
		tfstd = mean_std/np.sqrt(fint*tint)
		chisq = np.sum(resid.real**2/tfstd**2) + np.sum(resid.imag**2/tfstd**2) #gerr.real
	else:
		chisq = np.sum(resid.real**2) + np.sum(resid.imag**2)

	return np.array([chisq, num_valid_eqs, n_valid_sols], dtype=float)

def akaike_info_criterion(neq, npa, chisq):
	with np.errstate(divide='ignore', invalid='ignore'):
		c_aic = 2*npa + neq*np.log(chisq/neq) + (2*npa**2 + npa)/(neq-npa-1) #add /neq
	return c_aic


def get_2D_avg_gains(gobs, t_int, f_int, tbase=1, fbase=1):
	"""return the gains averaged along the time and frequency axis"""

	n_tchunks, n_dir, n_tim, n_fre, n_ant, n_cor, n_cor = gobs.shape

	It = np.arange(0, n_tim, t_int)
	Ncars = len(It)

	If = np.arange(0, n_fre, f_int)
	Ncarsf = len(If)

	if t_int ==1 and f_int==1:
		return gobs
	elif t_int==tbase and f_int==fbase:
		return gobs
	else:
		outgains = np.zeros_like(gobs)
		for f in range(n_fre):
			rc = f//f_int
			if rc == Ncarsf-1:
				tmp = np.sum(gobs[:,:,:,If[rc]::,:,:,:], axis=3)
				df = n_fre - If[rc]
			else:
				tmp = np.sum(gobs[:,:,:,If[rc]:If[rc+1],:,:,:], axis=3)
				df = f_int

			for t in range(n_tim):
				rr = t//t_int
				if rr == Ncars-1:
					tmp2 = np.sum(tmp[:,:,It[rr]::,:,:,:], axis=2)
					dt = n_tim - It[rr]
				else:
					tmp2 = np.sum(tmp[:,:,It[rr]:It[rr+1],:,:,:], axis=2)
					dt = t_int

				outgains[:,:,t,f,:,:,:] = tmp2/(dt*df)

	return outgains


def get_avg_gains(gobs, t_int, timeaxis=True):
	"""returns the gains average along the time axis for the given interval"""

	n_tchunks, n_dir, n_tim, n_fre, n_ant, n_cor, n_cor = gobs.shape

	if timeaxis:
		I = np.arange(0, n_tim, t_int)
		Ncars = len(I)

		if t_int == 1:
			return gobs
		else:
			outgains = np.zeros_like(gobs)
			for t in range(n_tim):
				rr = t//t_int
				if rr == Ncars-1:
					outgains[:,:,t,:,:,:,:] = np.average(gobs[:,:,I[rr]::,:,:,:,:], axis=2)
				else:
					outgains[:,:,t,:,:,:,:] = np.average(gobs[:,:,I[rr]:I[rr+1],:,:,:,:], axis=2)

			return outgains
	else:
		I = np.arange(0, n_fre, t_int)
		Ncars = len(I)

		if t_int == 1:
			return gobs
		else:
			outgains = np.zeros_like(gobs)
			for f in range(n_fre):
				rr = f//t_int
				if rr == Ncars-1:
					outgains[:,:,:,f,:,:,:] = np.average(gobs[:,:,:,I[rr]::,:,:,:], axis=3)
				else:
					outgains[:,:,:,f,:,:,:] = np.average(gobs[:,:,:,I[rr]:I[rr+1],:,:,:], axis=3)

			return outgains

def g_and_jhjinv_from_soln(alpha, Sigmaalpha, Nt, Nv, nt, nv, addjhj=False):
	"""
	Routine to broadcast soln to full resolution. The variance is broadcast such that the average in a soln inetrval
	equals the original variance
	:param alpha: Mt x Mv x Na x Ncorr x Ncorr array holding solution 
	:param Sigmaalpha: Mt x Mv x Na x Ncorr x Ncorr soln array holding Cramer-Rao bounds on gain solutions
	:param Nt: number f times on full grid
	:param Nv: number of freqs on full grid
	:param nt: time interval width
	:param nv: freq interval width
	:return: 
	"""
	_, _, Na, Ncorr, _ = alpha.shape
	gobs = np.zeros((Nt, Nv, Na, Ncorr, Ncorr), dtype=np.complex128)
	jhjinv = np.zeros((Nt, Nv, Na, Ncorr, Ncorr), dtype=np.float64)
	mean_std = np.mean(Sigmaalpha[...,0,0].real)
	It = np.arange(0, Nt, nt)
	Mt = It.size
	Iv = np.arange(0, Nv, nv)
	Mv = Iv.size
	It = np.concatenate((It, np.array([Nt])))
	Iv = np.concatenate((Iv, np.array([Nv])))
	gobs, jhjinv, nmax = _g_and_jhjinv_from_soln_impl(gobs, jhjinv, alpha, Sigmaalpha, Mt, It, Mv, Iv, Na, Ncorr, mean_std)

	if addjhj and nmax>1:
		std = 0.25*np.sqrt(nmax)*mean_std
		gobs += std*np.random.normal(size=gobs.shape) + 1j*std*np.random.normal(size=gobs.shape)
		gobs[...,(0,1),(1,0)] = 0
		print(nmax, " Here is nmax guy and std ", std)

	return gobs, jhjinv, np.sqrt(nmax/2)*mean_std


# TOD0 - optimise axis ordering
@numba.jit(nopython=True, nogil=True, cache=True)
def _g_and_jhjinv_from_soln_impl(gobs, jhjinv, alpha, Sigmaalpha, Mt, It, Mv, Iv, Na, Ncorr, mean_std):
	nmax = 0
	for p in range(Na):
		for c in range(Ncorr):
			for i in range(Mt):
				for j in range(Mv):
					N = (It[i + 1] - It[i]) * (Iv[j + 1] - Iv[j]) #+ 0j # the number of terms in this soln interval
					if N>nmax:
						nmax = N
					for t in range(It[i], It[i + 1]):
						for v in range(Iv[j], Iv[j + 1]):
							gobs[t, v, p, c, c] = alpha[i, j, p, c, c]
							jhjinv[t, v, p, c, c] = mean_std*np.sqrt(N/2) #Sigmaalpha[i, j, p, c, c].real

	return gobs, jhjinv, nmax


def optimal_time_freq_interval_2D_same_gains(dbfile, msname, name, tint, fint, tchunk, verbosity=False, prefix="G", return_freq=True, usejhj=False):
	"""extract 2D AIC"""

	ujhj, ajhj = usejhj

	dbgains, freqs = extract_from_db(dbfile, name, return_freq)#[:,0:1,...]
	_, nfreq, _, _, _, = dbgains.shape

	try:
		dbjhj = extract_from_db(dbfile, name=name+'.err') #[:,:,...]
	except:
		dbjhj = extract_from_db(dbfile[:-4]+"_err.npy") #[:,:,...]

	# dbjhj = np.ones_like(dbgainserr, dtype=np.float64)
	
	Nt, Nf, Na, _, __, = dbgains.shape

	ntc = tchunk//tint
	
	#import pdb; pdb.set_trace()

	tints = list(range(1, ntc+1, 1)) 
	fints = list(range(1, Nf+1, 1))
	fints[0] = 1

	tfs = dict(t=tint*np.array(tints), f=fint*np.array(fints))

	gaic = np.zeros((len(fints), len(tints)))
	chisqs = np.zeros((len(fints), len(tints), 3))


	gobs = dbgains[np.newaxis, np.newaxis, ...]
	jhjinv = dbjhj[np.newaxis, np.newaxis, ...]

	for i, fi in enumerate(fints):
		for j, ti in enumerate(tints):
			if j%20==0:
				print("running, ", fi,j)
			avg_gains = get_2D_avg_gains(gobs, ti, fi)
			chisq, num_valid_eqs, n_valid_sols = compute_gains_chisq(gobs, avg_gains, ti, fi, jhjinv, usejhj=ujhj)
			chisqs[i,j] = chisq, num_valid_eqs, n_valid_sols

			gaic[i,j] = akaike_info_criterion(num_valid_eqs, n_valid_sols, chisq) #

	gaic[0,0] = gaic[0,1]


	fi, ti = np.unravel_index(np.argmin(gaic, axis=None), gaic.shape)
	fgrid, tgrid = fint*fints[fi], tint*tints[ti]

	# make a color map of fixed colors
	# cmap = colors.ListedColormap(['blue', 'red'])
	# bounds=np.linspace(0.8,1.2,50)

	LOGGER.info(f"Suggested optimal intervals: time-int = {tgrid}, freq-int = {fgrid}")
	
	norm=colors.PowerNorm(gamma=1./4) #colors.BoundaryNorm(bounds, cmap.N) #LinearSegmentedColormap

	fig = plt.figure()
	cs = plt.imshow(gaic/1e6, extent = [tint, tint*tints[-1], fint*fints[-1], fint], aspect='auto', cmap="jet", norm=norm) #/np.max(gaic) #/1e6 - 3 # , norm=norm
	
	xi, xy = cs.get_clim()
	cb = fig.colorbar(cs, ticks=[xi, xy]) #[:,10:]
	
	cb.set_label("AIC",size=25)

	plt.xlabel("Time interval",size=30)
	plt.ylabel("Frequency interval",size=30)
	plt.tick_params(labelsize=25)
	
	fig.suptitle("freq-int, time-int = (%d, %d)"%(fgrid, tgrid), fontsize=25, x=0.5, y=1.0)
	fig.tight_layout()
	
	outname = dbfile[:-3]+"-" + prefix
	np.save(outname+"-AIC.npy", gaic)
	np.save(outname+"-chisq.npy", chisqs)
	np.save(outname+"-grid.npy", tfs)

	plt.savefig(outname+"_cubi_AIC_2D.pdf")

	plt.clf()
	plt.close()

	return tgrid

def optimal_time_freq_interval_2D_gains(dbfile, msname, name, tint, fint, tchunk, verbosity=False, prefix="G", return_freq=True, usejhj=False):
	"""extract 2D AIC"""

	dbgains2, freqs = extract_from_db(dbfile, name, return_freq)#[:,0:1,...]
	_, nfreq, _, _, _, = dbgains2.shape

	dbgains = dbgains2
	# print(dbgains.shape, "shape")

	try:
		dbjhj = extract_from_db(dbfile, name=name+'.err') #[:,:,...]
	except:
		dbjhj = extract_from_db(dbfile[:-4]+"_err.npy") #[:,:,...]

	# dbjhj = np.ones_like(dbgainserr, dtype=np.float64)
	
	Nt, Nf, Na, _, __, = dbgains.shape

	ntc = tchunk #int(np.min(ntchunks))  #64 #151#tchunk #
	# print("ntc is {}, scan_ids {}, ntchunks".format(ntc, scan_ids, ntchunks))
	
	#import pdb; pdb.set_trace()

	Nf = 1012

	# tints = np.array(range(1, ntc+1, 1))
	# fints = np.array([8]) #np.array(range(1, Nf+1, Nf)) # 1, 64, 512

	tints = list(range(0, 151, 6)) #0, 151, 6
	tints[0] = 1
	tints[-1] = 151
	fints = list(range(0, 512, 16))
	fints[0] = 1

	# fints = [32, 64]

	# tints = np.insert(tints, 0, 11)
	# fints = np.insert(fints, 0, 1)
	
	gaic = np.zeros((len(fints), len(tints)))
	chisqs = np.zeros((len(fints), len(tints), 3))

	ujhj, ajhj = usejhj

	f_int = fint #1 #Nf for reasons to explain later
	gobs, jhjinv, mean_std = g_and_jhjinv_from_soln(dbgains, dbjhj, Nt, Nf, tint, f_int, addjhj=ajhj) #Nf -> 1024 #
	
	# print(gobs.shape)
	# np.save(dbfile[:-3]+"-" + prefix+"-gobs.npy", gobs)
	# sys.exit()

	gobs = gobs[np.newaxis, np.newaxis, ...]
	jhjinv = jhjinv[np.newaxis, np.newaxis, ...]

	for i, fi in enumerate(fints):
		for j, ti in enumerate(tints):
			if j%20==0:
				print("running, ", fi,j)
			avg_gains = get_2D_avg_gains(gobs, ti, fi)
			chisq, num_valid_eqs, n_valid_sols = compute_gains_chisq(gobs, avg_gains, ti, fi, jhjinv, usejhj=ujhj)
			chisqs[i,j] = chisq, num_valid_eqs, n_valid_sols

			gaic[i,j] = akaike_info_criterion(num_valid_eqs, n_valid_sols, chisq) #

	fc = fints.index(fint) + 1
	for i in range(tint):
		gaic[0:fc,i] = gaic[0,tint]

	for i, fi in enumerate(fints[0:fc]):
		tc = fint//fi
		gaic[i, 0:tc] = gaic[0,tint]

	fi, ti = np.unravel_index(np.argmin(gaic, axis=None), gaic.shape)
	fgrid, tgrid = fints[fi], tints[ti]

	# make a color map of fixed colors
	# cmap = colors.ListedColormap(['blue', 'red'])
	# bounds=np.linspace(0.8,1.2,50)

	LOGGER.info(f"Suggested optimal intervals: time-int = {tgrid}, freq-int = {fgrid}")
	
	norm=colors.PowerNorm(gamma=1./4) #colors.BoundaryNorm(bounds, cmap.N) #LinearSegmentedColormap

	fig = plt.figure()
	cs = plt.imshow(gaic/1e6, extent = [1, tints[-1], fints[-1], 1], aspect='auto', cmap="jet") #/np.max(gaic) #/1e6 - 3 # , norm=norm
	
	xi, xy = cs.get_clim()
	cb = fig.colorbar(cs, ticks=[xi, xy]) #[:,10:]
	
	cb.set_label("AIC",size=25)

	plt.xlabel("Time interval",size=30)
	plt.ylabel("Frequency interval",size=30)
	plt.tick_params(labelsize=25)
	
	# fig.suptitle("${\Delta tf}$ = (%d, %d)"%(tgrid, fgrid*fint), fontsize=10, x=0.5, y=1)
	fig.tight_layout()
	
	outname = dbfile[:-3]+"-" + prefix
	np.save(outname+"-AIC.npy", gaic)
	np.save(outname+"-gobs.npy", gobs)
	np.save(outname+"-chisq.npy", chisqs)

	plt.savefig(outname+"_cubi_AIC_2D.pdf")

	plt.clf()
	plt.close()

	return tgrid

def optimal_time_freq_interval_from_gains(dbfile, msname, name, tint, fint, tchunk, verbosity=False, prefix="G", return_freq=True, usejhj=False):

	dbgains2, freqs = extract_from_db(dbfile, name, return_freq)#[:,0:1,...]
	_, nfreq, _, _, _, = dbgains2.shape

	dbgains = dbgains2
	# print(dbgains.shape, "shape")
	
	try:
		dbjhj = extract_from_db(dbfile, name=name+'.err') #[:,:,...]
	except:
		dbjhj = extract_from_db(dbfile[:-4]+"_err.npy") #[:,:,...]
	
	# rms, min_err, max_err = np.mean(dbgainserr[..., (0,1), (0,1)] + 1e-12), np.min(dbgainserr[..., (0,1), (0,1)] + 1e-12),\
					# np.max(dbgainserr[..., (0,1), (0,1)] + 1e-12)

	# if verbosity:
		# print("Gains rms: mean, min, max", rms, min_err, max_err)

	# dbjhj = np.ones_like(dbgainserr, dtype=np.float64)
	
	Nt, Nf, Na, _, __, = dbgains.shape

	# t = table(msname, ack=False)
	# time_col = t.getcol("TIME")
	# scans = t.getcol("SCAN_NUMBER")
	# scan_ids = np.unique(scans)
	# t.close()
		
	# ntchunks = np.zeros(len(scan_ids))

	# for i, scan_id in enumerate(scan_ids):
	# 	ntchunks[i] = len(np.unique(time_col[scans==scan_id]))

	ntc = tchunk #int(np.min(ntchunks))  #64 #151#tchunk #
	# print("ntc is {}, scan_ids {}, ntchunks".format(ntc, scan_ids, ntchunks))
	
	#import pdb; pdb.set_trace()

	tints = np.array(range(1, ntc+1, 1))
	fints = np.array(range(1, Nf+1, 1))

	# tints = np.insert(tints, 0, 11)
	# fints = np.insert(fints, 0, 1)
	
	gaic = np.zeros((len(fints), len(tints)))
	chisqs = np.zeros((len(fints), len(tints), 3))
	ujhj, ajhj = usejhj

	f_int = 1 #Nf for reasons to explain later
	gobs, jhjinv, mean_std = g_and_jhjinv_from_soln(dbgains, dbjhj, Nt, Nf, tint, f_int)
	gobs = gobs[np.newaxis, np.newaxis, ...]
	jhjinv = jhjinv[np.newaxis, np.newaxis, ...]

	for i, fi in enumerate(fints):
		for j, ti in enumerate(tints):
			avg_gains = get_2D_avg_gains(gobs[:,:,:,fi-1:fi,...], ti, 1)
			chisq, num_valid_eqs, n_valid_sols = compute_gains_chisq(gobs[:,:,:,fi-1:fi,...], avg_gains, ti, fi, jhjinv[:,:,:,fi-1:fi,...], usejhj=ujhj) # , jhjinv[:,:,:,fi-1:fi,...]

			chisqs[i,j] = chisq, num_valid_eqs, n_valid_sols

			gaic[i,j] = akaike_info_criterion(num_valid_eqs, n_valid_sols, chisq+1e-12)

		LOGGER.debug("Done for frequency block {}/{}".format(i+1, Nf))

	for i in range(tint):
		gaic[:,i] = gaic[:,tint]

	tmins = tints[np.argmin(gaic, axis=1)]
	ta = max(np.min(tmins), tint)

	LOGGER.info(f"Suggested optimal time interval is {ta}")

	fig, ax1 = plt.subplots()

	temp = 0.80
	extra_ys = 2
	right_additive = (0.98-temp)/float(extra_ys)

	ax1.set_xlabel("Time-interval", size=30)

	lns = None

	mod_val = 2 if len(fints) > 8 else 1

	for i, fi in enumerate(fints):
		# print(fi, len(tints), gaic.shape)
		ln = ax1.plot(tints[ta//3:], gaic[i, ta//3:]/np.min(gaic[i, :]), linestyle='-', label = "%.2f GHz"%freqs[i], linewidth=2) #/np.min(gaic[i, 3:]) #ta//3:

		if lns == None:
			lns = ln
		elif (i+1)%mod_val ==0:
			lns += ln
		else:
			pass

	ax1.set_ylabel("AIC")
	ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

	labs = [l.get_label() for l in lns]
	ax1.legend(lns, labs, loc='best',fontsize = 'small')
	
	fig.tight_layout()

	outname = dbfile[:-4] + "-" + prefix

	plt.savefig(outname+"_cubi_AIC_freq.pdf")
	
	plt.clf()
	plt.close()

	np.save(outname+"-AIC.npy", gaic)
	np.save(outname+"-gobs.npy", gobs)
	np.save(outname+"-chisq.npy", chisqs)

	return ta