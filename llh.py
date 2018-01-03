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

from scipy.optimize import minimize

import scipy.optimize
import scipy as scp

# ------------------------------- Settings ---------------------------- #

nugen_path = 'data/GFU/SplineMPEmax.MuEx.MC.npy'
GFU_path = 'data/GFU/SplineMPEmax.MuEx.IC86-2016.npy'
# use only Ibc for now (later use all!)
LCC_path = "/Users/afrancko/Optical/SEDmWhitePaper/LC_Uli/lcs_strawman_msip6m_Ibc_nugent_detected_real_y2016.pkl"

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
            'maxDist':np.deg2rad(3),
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
 
    dist = GreatCircleDistance(nuData['ra'],
                               nuData['dec'],
                               ra, dec)
    mask = dist < settings['maxDist']

    nuDataClose = nuData[mask]
        
    if len(nuDataClose) == 0:
        print "no neutrino found within %.1f degrees"%(np.rad2deg(settings['maxDist']))

    # remove neutrino that are more than 100 days from t0
    maskT0 =  abs(nuDataClose['time'] - t0)>5.
    maskTmax = ((nuDataClose['time']>(tmax-5)) & (nuDataClose['time']<(tmax+30)))
    #mask = abs(nuDataClose['time'] - t0)>5. or ((nuDataClose['time'])>(tmax-5) and (nuDataClose['time'])<(tmax+30))

    nuDataClose = nuDataClose[maskT0 | maskTmax]

    if len(nuDataClose) == 0:
        print "no neutrino found within 100 days from t0=%f"%t0

    return nuDataClose
   


def negTS(ns, S, B):
 
    N = float(len(S))

    X = 1./N * (S/B-1)
    ts = 2*np.sum(np.log(ns*X+1))
    
    #llh = np.sum(np.log(ns/N*S + (1.-(ns/N))*B))
    #llh0 = np.sum(np.log(B))
    #ts = 2*(llh-llh0)
    
    print "ns, TS ", ns, ts
    return -ts


def TS(ra, dec, t0, lc, nuData):
    
    tmax = lc['time'][np.argmax(lc['flux'])]

    #print "t0 ", t0
    #print "tmax ", tmax
    #print ra, dec
    
    nu = get_neutrinos(ra, dec, t0, tmax, nuData)

    if len(nu)==0:
        print "no neutrinos found."
    
    coszen = np.cos(utils.dec_to_zen(nu['dec']))
    B = (10 ** (coszen_spline(coszen))) /(2. * np.pi) 
   
    coszenS = np.cos(utils.dec_to_zen(dec))
    acceptance = 10**coszen_signal_reco_spline(coszenS)

    S = 1./(2.*np.pi*nu['sigma']**2)*np.exp(-GreatCircleDistance(ra, dec, nu['ra'], nu['dec'])**2 / (2.*nu['sigma']**2)) * acceptance 

    print "dist ", np.rad2deg(GreatCircleDistance(ra, dec, nu['ra'], nu['dec']))
    print "sigma ", np.rad2deg(nu['sigma'])
    print "1/(2pi sig**2) ", 1./(2.*np.pi*nu['sigma']**2)
    
    print "S ", S
    print "B ", B
    print "S/B ", S/B
    print "exp ", np.exp(-GreatCircleDistance(ra, dec, nu['ra'], nu['dec'])**2 / (2.*nu['sigma']**2))
    
    print "acceptance ", acceptance

    print "len(nu) ", len(nu)
    
    bounds = [(0.,len(nu))]

    #S = S*0.0

    #print "S ", S
    
    #res = minimize(negLogLike,x0=0,
    #               args=(S,B),
    #               method='SLSQP',
    #               #method='Nelder-Mead',#'SLSQP',
    #               bounds=bounds, options={'maxiter':100,'disp':False,'ftol':1e-8})
    
    x,f,d = scp.optimize.fmin_l_bfgs_b(negTS, 0.0, args=(S,B), bounds=bounds, approx_grad=True, epsilon=0.1)

    print d
    
    print d['warnflag'] 
    
    nsMax = x #res.x

    print 'nsMax ', nsMax
    
    ts = -negTS(nsMax,S,B)

    narray = np.linspace(0,len(nu),100)
    for n in narray:
        tsn = -negTS(n,S,B)
    
    print('TS: {} \n'.format(ts))
         
    return ts


def simulate(lc, nuData):
    i = 0
    
    ts = TS(np.deg2rad(lc['meta']['ra'][i]), np.deg2rad(lc['meta']['dec'][i]),
            lc['meta']['t0'][i],lc['lcs'][i], nuData)
    print ts
    
    
if __name__ == '__main__':

    jobN = int(sys.argv[1])
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
    np.random.shuffle(nuData['ra'])

    print 'splinename', spline_name
   
    if 1:#not os.path.exists('coszen_spl%s.npy'%spline_name) or \
        #not os.path.exists('E_spline.npy%s'%spline_name) or \
        #not os.path.exists('coszen_signal_spl%s.npy'%spline_name):
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
        llh_bg_dist= simulate(lc, nuData)#, settings['Nsim'], filename=filename)
    else:
        print('Load Trials...')
        llh_bg_dist = np.load(filename)

