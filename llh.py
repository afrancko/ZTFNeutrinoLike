#! /usr/bin/env python
# coding: utf-8

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from astropy.coordinates import SkyCoord
# from astropy import units as u
# from plot_conf import *

# from fancy_plot import *
import numpy as np

# from scipy.interpolate import interp1d
# import pyfits as fits
#from astropy.table import Column
import sys
from numpy.lib.recfunctions import rec_append_fields
# from scipy.interpolate import interp2d, InterpolatedUnivariateSpline, RectBivariateSpline
import os
import utils
import pickle

# from scipy.optimize import minimize
#
# import scipy.optimize
import scipy as scp

# import numpy.lib.recfunctions as rfn

# ------------------------------- Settings ---------------------------- #

source = "/afs/ifh.de/user/s/steinrob/scratch/ZTF_neutrino/"

nugen_path = source + 'data/GFU/SplineMPEmax.MuEx.MC.npy'
GFU_path = source + 'data/GFU/SplineMPEmax.MuEx.IC86-2016.npy'
# use only Ibc for now (later use all!)
LCC_path = source + "data/lcs_strawman_msip6m_Ibc_nugent_detected_real_y2016" \
                    ".pkl"

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

addinfo = 'test'

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

    #print "N events ", len(nuData)
    
    nuDataClose = nuData[mask]

    #print "After space cut ", len(nuDataClose)
    
    if len(nuDataClose) == 0:
        print "no neutrino found within %.1f degrees"%(np.rad2deg(settings['maxDist']))

    # remove neutrino that are more than 100 days from t0
    maskT0 =  abs(nuDataClose['time'] - t0)<5.
    maskTmax = ((nuDataClose['time']>(tmax-5)) & (nuDataClose['time']<(tmax+30)))
    #mask = abs(nuDataClose['time'] - t0)>5. or ((nuDataClose['time'])>(tmax-5) and (nuDataClose['time'])<(tmax+30))

    nuDataClose = nuDataClose[maskT0]# | maskTmax]

    #print "After time cut ", len(nuDataClose)
    
    if len(nuDataClose) == 0:
        print "no neutrino found within 100 days from t0=%f"%t0

    return nuDataClose
   


def negTS(ns, S, B):
 
    N = float(len(S))
    
    #if ns>=N:
    #    return 1e6
    
    llh = np.sum(np.log(ns/N*S + (1.-(ns/N))*B))
    llh0 = np.sum(np.log(B))
    ts = 2*(llh-llh0)

    return -ts


def TS(ra, dec, t0, lc, nuData):
    
    tmax = lc['time'][np.argmax(lc['flux'])]
    
    nu = get_neutrinos(ra, dec, t0, tmax, nuData)

    if len(nu)==0:
        print "no neutrinos found."
    
    coszen = np.cos(utils.dec_to_zen(nu['dec']))

    # devide by 2 because we're only looking at 0.5 years of light curve data? Spline is derived from 1y of data?
    B = (10 ** (coszen_spline(coszen)))  / (2 * np.pi) / 2.

    #print B, np.sum(B)
    
    coszenS = np.cos(utils.dec_to_zen(dec))
    acceptance = 10**coszen_signal_reco_spline(coszenS)

    S = 1./(2.*np.pi*nu['sigma']**2)*np.exp(-GreatCircleDistance(ra, dec, nu['ra'], nu['dec'])**2 / (2.*nu['sigma']**2)) * acceptance 

    #plt.figure()
    #plt.plot(nu['time'],np.log10(S/B),'ob')
    #plt.ylim(-10,10)
    #plt.savefig('plots/SoB_vs_time.png')
    
    bounds = [(0.,len(nu))]
    
    #res = minimize(negLogLike,x0=0,
    #               args=(S,B),
    #               method='SLSQP',
    #               #method='Nelder-Mead',#'SLSQP',
    #               bounds=bounds, options={'maxiter':100,'disp':False,'ftol':1e-8})
    
    x,f,d = scp.optimize.fmin_l_bfgs_b(negTS, 0.1, args=(S,B), bounds=bounds, approx_grad=True, epsilon=1e-8)

    #print d
    
    #print d['warnflag'] 
    
    nsMax = x #res.x

    #print 'nsMax ', nsMax
    
    ts = -negTS(nsMax,S,B)

    #print 'tsMax ', ts

    #print "N ", len(S)
    
    narray = np.linspace(0,min(300,len(S)-2),100)
    tsArray = []
    for n in narray:
        tsn = -negTS(n,S,B)
        tsArray.append(tsn)

    # plt.figure()
    # plt.plot(narray,tsArray)
    # plt.xlabel('ns')
    # plt.ylabel('TS')
    # plt.savefig('plots/ns_vs_TS.png')
    # plt.close()
    
    #print('TS: {} \n'.format(ts))
         
    return ts



def inject(lc,nuSignal,gamma, Nsim, dtype):

    i = 0
    
    raS = np.deg2rad(lc['meta']['ra'][i])
    decS = np.deg2rad(lc['meta']['dec'][i])
    t0 = lc['meta']['t0'][i]

    
    enSim = []
    sigmaSim = []
    zenSim = []
    aziSim = []
    raSim = []
    timeSim = []
    distTrue = []
    weightSim = []
    
    # check if single source or source list
    if type(raS) is np.float64:
        zen_mask = np.abs(np.cos(nuSignal['zenith'])-np.cos(utils.dec_to_zen(decS)))<0.01
        fSource = nuSignal[zen_mask]
        
        print "selected %i events in given zenith range"%len(fSource)
        for i in range(len(fSource)):
            rotatedRa, rotatedDec = utils.rotate(fSource['azimuth'][i],
                                                 utils.zen_to_dec(fSource['zenith'][i]),
                                                 raS, decS, 
                                                 fSource[settings['az_reco']][i],
                                                 utils.zen_to_dec(fSource[settings['zen_reco']][i]))
            fSource[i][settings['az_reco']] = rotatedRa
            fSource[i][settings['zen_reco']] = utils.dec_to_zen(rotatedDec)

        weight = fSource['ow']/10**fSource['logE']**gamma
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
        distTrue.append(GreatCircleDistance(rotatedRa, rotatedDec, raS, decS))

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

    print "finished injection"
    
    return sim

        
def simulate(lc, nuData):
    i = 0

    ts = []
    
    for i in range(len(lc['meta']['ra'])):
        ts.append(TS(np.deg2rad(lc['meta']['ra'][i]), np.deg2rad(lc['meta']['dec'][i]),
                     lc['meta']['t0'][i],lc['lcs'][i], nuData))
    return ts
    
    
if __name__ == '__main__':

    jobN = int(sys.argv[1])
    nSig = int(sys.argv[2])
    # get Data
    lc = readLCCat()
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
    
    # scramble ra
    #np.random.shuffle(nuData['ra'])

    # convert from MJD to JD to be consistent with LC time format, add 1 year to make sure neutrino data and LC overlap in time
    nuData['time'] = nuData['time']+2400000.5+365.*1.5
    
    
    # inject some signal events
    if nSig>0:
        print "inject %i signal events "%nSig
        simEvents = inject(lc,nuDataSig, settings['gamma'], nSig, nuData.dtype)
        # merge data (=background) with injected signal events


        #data_all = simEvents
        
        data_all = nuData.copy()
        print "data_all ", len(data_all)
        data_all.resize(len(nuData) + len(simEvents))
        data_all[len(nuData):] = simEvents
        print "data_all after merging ", len(data_all)

    else:
        data_all = nuData

    print 'splinename', spline_name
   
    if not (os.path.exists('coszen_spl%s.npy'%spline_name) and
         os.path.exists('E_spline.npy%s'%spline_name) and
         os.path.exists('coszen_signal_spl%s.npy'%spline_name)):
            print('Create New Splines..')
            utils.create_splines(nuData,nuDataSig,
                                 settings['zen_reco'],
                                 settings['az_reco'], 
                                 settings['E_reco'], spline_name)
    E_spline = np.load('E_spline%s.npy'%spline_name)[()]
    coszen_spline = np.load('coszen_spl%s.npy'%spline_name)[()]
    coszen_signal_spline = np.load('coszen_signal_spl%s.npy'%spline_name)[()]
    coszen_signal_reco_spline = np.load('coszen_signal_reco_spl%s.npy'%spline_name)[()]

    print('Generating PDFs..Finished')

    filename = './output/{}_llh_{}_{:.2f}_{}.npy'.format(addinfo, settings['Nsim'],
                                              settings['gamma'],
                                              jobN)

    print('##############Create BG TS Distrbution##############')
    if 1:#not os.path.exists(filename):
        llh_bg_dist= simulate(lc, data_all)#, settings['Nsim'], filename=filename)
        np.save(filename,llh_bg_dist)
    else:
        print('Load Trials...')
        llh_bg_dist = np.load(filename)

