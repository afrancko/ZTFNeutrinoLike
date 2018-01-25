#! /usr/bin/env python
# coding: utf-8


# from astropy.coordinates import SkyCoord
# from astropy import units as u
# from plot_conf import *

from fancy_plot import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pyfits as fits
#from astropy.table import Column
import sys
from numpy.lib.recfunctions import rec_append_fields
from scipy.interpolate import interp2d, InterpolatedUnivariateSpline, RectBivariateSpline
import os
import utils
import pickle
from scipy.stats import norm
from scipy.optimize import minimize

import scipy.optimize
import scipy as scp

import numpy.lib.recfunctions as rfn

PLOT = False

# ------------------------------- Settings ---------------------------- #

nugen_path = 'data/GFU/SplineMPEmax.MuEx.MC.npy'
GFU_path = 'data/GFU/SplineMPEmax.MuEx.IC86-2016.npy'
# use only Ibc for now (later use all!)
LCC_path = "data/lcs_strawman_msip6m_all_real_y2016_vs02.pkl"

settings = {'E_reco': 'logE',#'muex',
            'zen_reco': 'zenith',
            'az_reco': 'azimuth',
            'sigma': 'sigma',
            'time': 'time',
            'gamma': 2.1,
            'dec_true' : 'trueDec',
            'ra_true' : 'trueRA',
            'ra_reco': 'ra',
            'dec_reco': 'dec',
            'ftypes': ['weight'],
            #'ftypes': ['Conventional', 'Prompt', 'astro'],  # atmo = conv..sry for that
            #'ftype_muon': 'GaisserH3a', #???????
            'Nsim': 50000,
            'Phi0': 0.91,
            'maxDist':np.deg2rad(1800),
            'E_weights': True} 

FIX_POS = True
DEC = 5

addinfo = 'test'

if FIX_POS:
    addinfo += '_fix_pos'


spline_name = 'spline'

# --------------------------------------------------------------------------- #

@np.vectorize
def powerlaw(trueE, ow):
    return ow * settings['Phi0'] * 1e-18 * (trueE * 1e-5) ** (- settings['gamma'])

    
def GreatCircleDistance(ra_1, dec_1, ra_2, dec_2):
    '''Compute the great circle distance between two events'''
    delta_dec = np.abs(dec_1 - dec_2)
    delta_ra = np.abs(ra_1 - ra_2)
    x = (np.sin(delta_dec / 2.))**2. + np.cos(dec_1) *\
        np.cos(dec_2) * (np.sin(delta_ra / 2.))**2.
    return 2. * np.arcsin(np.sqrt(x))


def readLCCat():
    return pickle.load( open( LCC_path, "rb" ) )


# get neutrinos close to source
def get_neutrinos(ra, dec, t0, tmax, nuData):

    #print t0, tmax
    
    dist = GreatCircleDistance(nuData['ra'],
                               nuData['dec'],
                               ra, dec)
    
    mask = dist < settings['maxDist']
    
    nuDataClose = nuData[mask]

    
    if len(nuDataClose) == 0:
        print "no neutrino found within %.1f degrees"%(np.rad2deg(settings['maxDist']))

    # remove neutrino that are more than 100 days from t0
    maskT0 =  abs(nuDataClose['time'] - t0)<5.
    #maskTmax = ((nuDataClose['time']>(tmax-5)) & (nuDataClose['time']<(tmax+30)))
    maskTmax = ((nuDataClose['time']>(t0-5)) & (nuDataClose['time']<(t0+90)))

    #nuDataClose = nuDataClose[maskT0]# | maskTmax]
    nuDataClose = nuDataClose[maskTmax]

    
    if len(nuDataClose) == 0:
        print "no neutrino found within 100 days from t0=%f"%t0

    return nuDataClose
   


def negTS(ns, X, S):
 
    ts = 2*np.sum(np.log(ns*X+1))
    
    return -ts


def TS(ra, dec, t0, z, lci, nuData, i):
    
    tmax = lci['time'][np.argmax(lci['flux'])]
    
    nu = get_neutrinos(ra, dec, t0, tmax, nuData)
    
    if len(nu)==0:
        print "no neutrinos found."
    
    coszen = np.cos(utils.dec_to_zen(nu['dec']))

    B = (10 ** (coszen_spline(coszen)))  / (2 * np.pi) 
    
    coszenS = np.cos(utils.dec_to_zen(dec))
    acceptance = 10**coszen_signal_reco_spline(coszenS)

    E_ratio = E_spline(coszen, nu['logE'],grid=False)
   
    if i%500 == 0:
        print "i ", i
    
    S = 1./(2.*np.pi*nu['sigma']**2)*np.exp(-GreatCircleDistance(ra, dec, nu['ra'], nu['dec'])**2 / (2.*nu['sigma']**2)) #* acceptance

    SoB = S/B #* E_ratio #z_spline(z) *
    
    X = 1./float(len(S)) * (SoB - 1)
    
    bounds = [(0.,len(nu))]
    #bounds = [(-len(nu),len(nu))]

    x,f,d = scp.optimize.fmin_l_bfgs_b(negTS, 0.1, args=(X,S), bounds=bounds, approx_grad=True, epsilon=1e-8)

    #print "warnflags ", d['warnflag'], d['task'] 

    if d['warnflag']>0:
        print "repeat fitting"
        x,f,d = scp.optimize.fmin_l_bfgs_b(negTS, 3, args=(X,S), bounds=bounds, approx_grad=True, epsilon=1e-8)
        print "warnflags ", d['warnflag'], d['task'] 
        if d['warnflag']>0:
            print "repeat fitting again"
            x,f,d = scp.optimize.fmin_l_bfgs_b(negTS, 3, args=(X,S), bounds=bounds, approx_grad=True, epsilon=1e-9)
            print "warnflags ", d['warnflag'], d['task'] 

    nsMax = x #res.x

    #print "nsMax ", nsMax
    
    ts = -negTS(nsMax,X,S)

    narray = np.linspace(0,min(20,len(S)-2),100)
    tsArray = []
    for n in narray:
        tsn = -negTS(n,X,S)
        tsArray.append(tsn)

    if PLOT:
        fig = plt.figure()
        plt.plot(narray,tsArray)
        plt.xlabel('ns')
        plt.ylabel('TS')
        plt.savefig('plots/ns_vs_TS_%i.png'%i)
        plt.close(fig)
    
    #print('TS: {} \n'.format(ts))
         
    return ts, nsMax



def inject(lc,i, nuSignal,gamma, Nsim, dtype):

    SNType = lc['type'][i]

    
    raS = np.deg2rad(lc['ra'][i])
    decS = np.deg2rad(lc['dec'][i])
    t0 = lc['t0'][i]

    SNid = lc['ID'][i]
    
    fname = 'SN_signal/neutrinos_%s_%i.npy'%(SNType,SNid)

    if FIX_POS:
        fname = 'SN_signal/SN_neutrinos_fixedPos_dec%i.npy'%DEC
    
    enSim = []
    sigmaSim = []
    zenSim = []
    aziSim = []
    raSim = []
    timeSim = []
    weightSim = []

    if os.path.exists(fname):
        print "load file %s"%fname
        fSource = np.load(fname)
    else:
        print "create file %s"%fname
        zen_mask = np.abs(np.cos(nuSignal['zenith'])-np.cos(utils.dec_to_zen(decS)))<0.005
        fSource = nuSignal[zen_mask]
        
        print "selected %i events in given zenith range"%len(fSource)
        for i in range(len(fSource)):
            rotatedRa, rotatedDec = utils.rotate(fSource['trueAzimuth'][i],
                                                 utils.zen_to_dec(fSource['trueZenith'][i]),
                                                 raS, decS, 
                                                 fSource[settings['az_reco']][i],
                                                 utils.zen_to_dec(fSource[settings['zen_reco']][i]))
            #print np.rad2deg(rotatedRa), np.rad2deg(rotatedDec)
            fSource[i][settings['az_reco']] = rotatedRa
            fSource[i][settings['zen_reco']] = utils.dec_to_zen(rotatedDec)
            
            
        np.save(fname,fSource)
            
            
    weight = fSource['ow']/fSource['trueE']**gamma
    draw = np.random.choice(range(len(fSource)),
                            Nsim,
                            p=weight / np.sum(weight))

    enSim.extend(fSource[draw][settings['E_reco']])
    sigmaSim.extend(fSource[draw][settings['sigma']])
    zenSim.extend(fSource[draw][settings['zen_reco']])
    aziSim.extend(fSource[draw][settings['az_reco']])
    raSim.extend(fSource[draw][settings['az_reco']])
    weightSim.extend(weight[draw])

    # have all neutrinos arrive at t0
    timeSim = np.ones_like(np.asarray(enSim))*t0
    #distTrue.append(GreatCircleDistance(rotatedRa, rotatedDec, raS, decS))

    # produce similar output to data file
    sim = dict()
    sim['logE'] = np.array(enSim)
    sim['ra'] =  np.array(raSim)
    sim['dec'] = utils.zen_to_dec(np.array(zenSim))
    sim['sigma'] = np.array(sigmaSim)
    sim['time'] = np.array(timeSim)
    sim['zenith'] = np.array(zenSim)
    sim['azimuth'] = np.array(zenSim)
    # fill dummies for run and event number
    sim['Run'] = np.zeros_like(zenSim)
    sim['Event'] = np.zeros_like(zenSim)
    sim['weight'] = np.zeros_like(weightSim)

    sim = np.array( zip(*[sim[ty] for ty in dtype.names]), dtype=dtype)
    
    return sim

        
def simulate(lc, nuData, Nlc, j=-1):

    ts = []
    nmax = []
    
    #print "number of light curves ", Nlc
    
    for i in range(Nlc):

        if j>=0:
            # don't shuffle data in case of signal injection (that will randomize the already injected signal)
            i=j
        else:
            np.random.shuffle(nuData['ra'])

        tsi = TS(np.deg2rad(lc['ra'][i]), np.deg2rad(lc['dec'][i]), 
                 lc['t0'][i], lc['z'][i], lc['lcs'][i], nuData, i)

        ts.append(tsi[0])
        nmax.append(tsi[1])

        if j>=0:
            break
        
    return ts, nmax
    
    
if __name__ == '__main__':

    jobN = int(sys.argv[1])
    nSig = int(sys.argv[2])
    # get Data
    lc = readLCCat()
    
    z_hist_BG = np.load('z_hist_BG.npy')
    lc['z'] = np.random.choice(z_hist_BG, size=len(lc['z']))

    if FIX_POS:
        lc['ra'] = np.zeros_like(lc['ra'])
        lc['dec'] = np.ones_like(lc['dec'])*DEC
        addinfo += '_dec%i'%DEC
        
    # to make things faster, just select a subset here
    Nlc = 500 #len(lc['ra'])
       
    nuData = np.load(GFU_path)
    weight = np.ones_like(nuData['ra'])
    nuData = rec_append_fields(nuData, 'weight',
                               weight,
                               dtypes=np.float64)
 
    
    nuDataSig = np.load(nugen_path)
    astro = powerlaw(nuDataSig['trueE'], nuDataSig['ow'])
    nuDataSig = rec_append_fields(nuDataSig, 'astro',
                                  astro,
                                  dtypes=np.float64)
    
    # convert from MJD to JD to be consistent with LC time format, add 1 year to make sure neutrino data and LC overlap in time
    nuData['time'] = nuData['time']+2400000.5+365.*1.5
     
    E_spline = np.load('E_spline%s.npy'%spline_name)[()]
    #E_spline = np.load('Filled_E_spline%s.npy'%spline_name)[()]
    #E_spline = np.load('Nearest_Neighbour_E_spline%s.npy'%spline_name)[()]

    
    coszen_spline = np.load('coszen_spl%s.npy'%spline_name)[()]
    coszen_signal_spline = np.load('coszen_signal_spl%s.npy'%spline_name)[()]
    coszen_signal_reco_spline = np.load('coszen_signal_reco_spl%s.npy'%spline_name)[()]

    z_spline = np.load('z_spl_20.0.npy')[()]
    
    print('Generating PDFs..Finished')

    filename = './output/{}_llh_{}_{:.2f}_{}.npy'.format(addinfo, nSig,
                                                         settings['gamma'],
                                                         jobN)

    filename_BG = './output/{}_llh_{}_{:.2f}_{}.npy'.format(addinfo, 0,
                                                            settings['gamma'],
                                                            jobN)


    print "filename ", filename
    
    if 1:#not os.path.exists(filename_BG):
        print('##############Create BG TS Distrbution##############')
        llh_bg_dist, nmax_dist = simulate(lc, nuData, Nlc)
        np.save(filename_BG,llh_bg_dist)
        np.save(filename_BG.replace('llh','ns'),nmax_dist)

        if PLOT:
            plt.figure()
            X2 = np.sort(llh_bg_dist)
            F2 = np.ones(len(llh_bg_dist)) - np.array(range(len(llh_bg_dist))) / float(len(llh_bg_dist))
            plt.plot(X2, F2)
            plt.xlabel('TS')
            plt.savefig('plots/TSDist_sig%i.png'%nSig)
    
    if nSig>0:
        Nlc = 500
        print "%i events will be injected on %i SNe"%(nSig,Nlc)

        z_hist_sig = np.load('z_hist_signal.npy')
        lc['z'] = np.random.choice(z_hist_sig, size=len(lc['z']))

        # inject some signal events
        print "inject %i signal events "%nSig
        print "len before merging  ", len(nuData)

        lc_Ibc = utils.selectSNType(lc,"Ibc")

        sig_ts_array = []
        sig_ns_array = []
            
        # inject the same amount of signal on each SN
        for i in range(min(Nlc,len(lc_Ibc['ra']))):
            np.random.shuffle(nuData['ra'])
            data_all = nuData
            
            simEvents = inject(lc_Ibc,i,nuDataSig, settings['gamma'], nSig, nuData.dtype)
            # merge data (=background) with injected signal events
            data_all = data_all.copy()
            old_size = len(data_all)
            data_all.resize(old_size + len(simEvents))
            data_all[old_size:] = simEvents
            #print "data_all after merging ", len(data_all)        
            
            sig_ts, sig_ns = simulate(lc, data_all, Nlc, i)
            
            sig_ts_array.append(sig_ts[0])
            sig_ns_array.append(sig_ns[0])
            
        print "save signal trails as ",filename
        np.save(filename,sig_ts_array)
        np.save(filename.replace('llh','ns'),sig_ns_array)
