# import packages (not all needed for simple case):

import matplotlib
matplotlib.use("TkAgg")


import matplotlib
matplotlib.use("Agg")
import pylab as py
import numpy as np
import scipy as sp
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astroquery.sdss import SDSS
import astropy.units as u

##############################################################################
# This is a simple prototype code for reading in a SAMI cube
#
# Example usage: 
# sami_readcube_simple.sami_readcube('36894_blue_7_Y13SAR1_P002_15T006.fits')
#
def sami_readcube(infile):

    # open file using pyfits:
    hdulist = fits.open(infile)

    # output the structure of the file:
    hdulist.info()

    # get data.  Assumes the data is the primary extension:
    cube = hdulist[0].data

    # get the variance vube:
    var = hdulist['VARIANCE'].data

    # get array sizes:
    (zs,ys,xs) = cube.shape
    print("input array sizes:",zs,"x",ys," x ",xs)
    print("1st python index is ",zs)
    print("2nd python index is ",ys)
    print("3rd python index is ",xs)

    # get wavelength info from header items in the primary header:
    # in this case the wavelength info is in the 3rd dimension,
    # which is the reverse of the way python reads the array:
    primary_header=hdulist['PRIMARY'].header
    crval1=primary_header['CRVAL3']
    cdelt1=primary_header['CDELT3']
    crpix1=primary_header['CRPIX3']
    naxis1=primary_header['NAXIS3']
    x=np.arange(naxis1)+1
    L0=crval1-crpix1*cdelt1 #Lc-pix*dL        
    lam=L0+x*cdelt1
    print('WCS info:',crval1,cdelt1,crpix1,naxis1)

    # simple plot of a slice:
    # define a figure
    fig1 = py.figure()
    # add a axis as a subplot (in this case just one):
    ax1 = fig1.add_subplot(1,1,1)
    #plot a slice:
    #cax1 = ax1.imshow(cube[1000,:,:],interpolation='nearest',origin='lower')
    mid = cube.shape[0] // 2
    cax1 = ax1.imshow(cube[mid,:,:], interpolation='nearest', origin='lower')

    cbar1 = fig1.colorbar(cax1,ax=ax1)
    ax1.set(xlabel='pixels',label='pixels')

    # second plot of median:
    fig2 = py.figure()
    # add a axis as a subplot (in this case just one):
    ax2 = fig2.add_subplot(1,1,1)
    # get median image (median over the 0th (wavelength) axis.
    # this funtion ignores NaN values:
    medim = np.nanmedian(cube,axis=0)
    #plot the median image:
    cax2 = ax2.imshow(medim,interpolation='nearest',origin='lower')
    cbar2 = fig2.colorbar(cax2,ax=ax2)
    ax2.set(xlabel='pixels',ylabel='pixels')
    
    # plot central spectrum (and a couple of others):    
    # second plot of median:
    fig3 = py.figure()
    # add a axis as a subplot (in this case just one):
    ax31 = fig3.add_subplot(3,1,1)
    # get median image (median over the 0th (wavelength) axis:
    xcent = int(xs/2)
    ycent = int(ys/2)
    spec1 = cube[:,xcent,ycent]
    #plot spectrum
    ax31.plot(lam,spec1)
    ax31.set(xlabel='Wavelength (Angstroms)',ylabel='flux (10-16 erg/s/cm2/A)')

    # plot zoom in between 4800 and 5500A:
    ax32 = fig3.add_subplot(3,1,2)
    #plot spectrum
    ax32.plot(lam,spec1)
    ax32.set(xlabel='Wavelength (Angstroms)',ylabel='flux (10-16 erg/s/cm2/A)')
    ax32.set(xlim=[4800,5500])

    
###############################################################################
# script to use astroquery to get SDSS spectra
# ra/dec in degrees
#
def get_sdss_spec_radec(ra,dec):

    # define an astropy coordinate object:
    coord = SkyCoord(ra,dec, unit="deg")
    print(coord)

    # query for spectrum:
    #try:
    spec = SDSS.get_spectra(coordinates=coord,data_release=19,radius= 5.0*u.arcsec)
    #except (ConnectionError) as err:
    #except:
    #print('ERROR getting spectrum')
    #return
        
    print('shape of spec returned:',np.shape(spec))
        
    # construct the file name if something if found:
    if (spec != None):

        # loop through recovered spectra:
        ns = 0
        for spec1 in spec:
            hdr = spec1[0].header
            mjd = hdr['MJD']
            plateid = hdr['PLATEID']
            fiberid = hdr['FIBERID']

            ns = ns +1

            fname = str(plateid)+'-'+str(mjd)+'-'+str(fiberid)+'.fits'
            print(fname)
                
            spec1.writeto(fname,overwrite=True)
    else:
        print('No spectrum found')

###################################################################################
# read and plot an SDSS spectrum
#
def plot_sdss_spec(infile):

    # get basic info on file:
    fits.info(infile)
    
    # opening file
    hdulist = fits.open(infile)
    print('opening SDSS spectrum: ',infile)
    # print header info for primary:
    print(repr(hdulist[0].header))
    # not all files have BUNIT
    #fluxunit = hdulist[0].header['BUNIT']
    # print header info for COADD extension:
    print('\n COADD extension header:\n')
    print(repr(hdulist['COADD'].header))
    # read spectrum from COADD extension:
    sdss_spec_table = hdulist['COADD'].data
    sdss_flux = sdss_spec_table['flux']
    sdss_loglam = sdss_spec_table['loglam']
    # SDSS spectra are in log wavelength bins, to convert to linear:
    sdss_lam = 10.0**sdss_loglam
    # close fits file:
    hdulist.close()

    # define parts of spectrum where data is likely to be good:
    idx = np.where((sdss_lam>3700))
    
    # make simple plot:
    fig1 = py.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot(sdss_lam[idx],sdss_flux[idx])
    #ax1.set(xlabel='Wavelength (Angstroms)',ylabel='Flux ('+fluxunit+')')
    ax1.set(xlabel='Wavelength (Angstroms)',ylabel='Flux')
    
    
#########################################################
# function to read a SAMI aperture spectrum:
#
def sami_read_apspec(hdulist,extname,doareacorr=False,bugfix=True):

    sami_flux_blue = hdulist[extname].data
    header = hdulist[extname].header
    crval1=header['CRVAL1']
    cdelt1=header['CDELT1']
    crpix1=header['CRPIX1']
    naxis1=header['NAXIS1']
    x=np.arange(naxis1)+1
    L0=crval1-crpix1*cdelt1 #Lc-pix*dL        
    sami_lam_blue=L0+x*cdelt1

    if (doareacorr):
        # fix bug in ap spec code:
        if (bugfix):
            areacorr=areacorr/2.0
        areacorr = header['AREACORR']
        sami_flux_blue = sami_flux_blue * areacorr

    return sami_flux_blue,sami_lam_blue
    
#########################################################
# function to read a SAMI 2D map (e.g. Halpha map):
#
# Example usage:
# sami_readcube_simple.sami_readmap('323854_A_Halpha_default_1-comp.fits')
# note that default error extension os Halpha error.
#
def sami_readmap(infile,errext='halpha_err'):

    # open file using pyfits:
    hdulist = fits.open(infile)

    # output the structure of the file:
    hdulist.info()

    # get data.  Assumes the data is the primary extension:
    # flxu map is actually a 3D data set, but just take the first slice:
    fluxmap = hdulist[0].data[0,:,:]

    # get the error map:
    fluxerr = hdulist[errext].data[0,:,:]

    # get array sizes:
    (ys,xs) = fluxmap.shape
    print("input array sizes:",ys," x ",xs)
    print("1st python index is ",ys)
    print("2nd python index is ",xs)

    # simple plot of a slice:
    # define a figure
    fig1 = py.figure()
    # add a axis as a subplot (in this case just one):
    ax1 = fig1.add_subplot(1,1,1)
    #plot a slice:
    cax1 = ax1.imshow(fluxmap[:,:],interpolation='nearest',origin='lower')
    cbar1 = fig1.colorbar(cax1,ax=ax1)
    ax1.set(xlabel='pixels',label='pixels')



if __name__ == "__main__":
    sami_readmap("99742_A_Halpha_default_1-comp.fits")
    py.savefig("sami_map.png", dpi=150, bbox_inches="tight")



