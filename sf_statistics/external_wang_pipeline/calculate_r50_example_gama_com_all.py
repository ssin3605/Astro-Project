#use to caculate bin cube
#coding:utf8
import numpy as np
from astropy.io import fits
import useful_functions as uf    #Contains a milion little functions that I have written. Many of these have probably been implimented elsewhere, but some have not.
#import binning_data   #My code for binning the SAMI data in a few differnt ways
import scipy.ndimage
import re
import time
from astropy.cosmology import WMAP7
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
import glob
import math
####################################################
########### POINTING TO THE DATA ###################
####################################################

filedir='/suphys/dwan9990/di'

#A file that contains lots of information about the galaxies
datafile='/suphys/dwan9990/di/list/gama/gamaout_sf_hbsn3_v0.12.txt'
#myfile= open(datafile)
#L=len(myfile.readlines())

#fitdir='/export/bill1/sami_data/data_products/v0.11/lzifu_newcont_fits/1_comp/'
#fitdir='/export/bill1/sami_data/data_products/DR2/lzifu_binned_fits/products_annular_1_comp/'
#fitdir='/export/bill1/sami_data/data_products/DR2/lzifu_binned_fits/products_sectors_1_comp/'
#datadir='/export/bill1/sami_data/data_releases/v0.11/'

datadir='/export/bill1/sami_data/data_releases/v0.12/cubes/'
fitdir='/export/bill1/sami_data/data_products/v0.12/LZIFU/lzifu_default_products/'



####################################################
########### Get information about the galaxies  ####
####################################################

outputtable = open(filedir+'/list/gama/gamaout_com_v0.12.txt','w')   #Create a file to write to
outputtable.write ("#CATID re PA ellip z g_i Mstar BCGCATID MassA CIndex_n2ha_s ha_EW_n_s SFR_n2ha_s CIndex_n2ha_m ha_EW_n_m SFR_n2ha_m CIndex_n2ha_l ha_EW_n_l SFR_n2ha_l  CIndex_o3hb_s ha_EW_o_s SFR_o3hb_s CIndex_o3hb_m  ha_EW_o_m SFR_o3hb_m  CIndex_o3hb_l ha_EW_o_l SFR_o3hb_l CIndex_s2ha_s ha_EW_s_s SFR_s2ha_s CIndex_s2ha_m  ha_EW_s_m SFR_s2ha_m CIndex_s2ha_l ha_EW_s_l SFR_s2ha_l galclass galclass_exclude seeing seeing_re \n")# This will write the header line to the file

#Get information for the galaxies from the data file
objID_in=np.nan_to_num(np.genfromtxt(datafile,usecols=(0),skip_header=1,dtype=int))
reff_in=np.nan_to_num(np.genfromtxt(datafile,usecols=(1),skip_header=1))
pa_in=np.nan_to_num(np.genfromtxt(datafile,usecols=(2),skip_header=1))
ellip_in=np.nan_to_num(np.genfromtxt(datafile,usecols=(3),skip_header=1))
z_in=np.nan_to_num(np.genfromtxt(datafile,usecols=(4),skip_header=1))
g_icolour_in=np.nan_to_num(np.genfromtxt(datafile,usecols=(5),skip_header=1))
Mstar_in=np.nan_to_num(np.genfromtxt(datafile,usecols=(6),skip_header=1))
BCGCATID_in=np.nan_to_num(np.genfromtxt(datafile,usecols=(7),skip_header=1))
MassA_in=np.nan_to_num(np.genfromtxt(datafile,usecols=(8),skip_header=1))
MassA_in=np.nan_to_num(np.genfromtxt(datafile,usecols=(8),skip_header=1))
galclass_in=np.nan_to_num(np.genfromtxt(datafile,usecols=(15),skip_header=1))
galclass_exclude_in=np.nan_to_num(np.genfromtxt(datafile,usecols=(16),skip_header=1))
#Note these redshifts are used for luminosity distance calculation so should probably be z_tonry



def get_x(x,y,target):
    '''
    Define linear interpolation for half-light radius analysis
    x=vector of x-values
    y=vector of y-values
    target=target y value for which to extract the x value
    '''
    ind=uf.find_nearest_index(y,target)
    if y[ind]<target:
        x1=ind
        x2=ind+1
    else:
        x1=ind-1
        x2=ind
    m=float(y[x2]-y[x1])/float(x[x2]-x[x1])
    b=y[x1]-m*x[x1]
    return float(target-b)/float(m)


##########################################################
######### Calculate radial profile information ###########
##########################################################

def radial(ID):
    '''
    Function that takes a single argument ID
    ID is the GAMA ID in the objID array defined above.
    '''
    #Get the data for each galaxy
    ind=objID_in.tolist().index(ID)
    objID=objID_in[ind]
    pa=pa_in[ind]*np.pi/180.0
    pa_ori=pa_in[ind]
    ellip=ellip_in[ind]
    reff=reff_in[ind]
    z=z_in[ind]
    g_icolour=g_icolour_in[ind]
    Mstar=Mstar_in[ind]
    BCGCATID=BCGCATID_in[ind]
    MassA=MassA_in[ind]
    galclass=galclass_in[ind]
    galclass_exclude=galclass_exclude_in[ind]
    galclass_ex=galclass_exclude
    gMINUSi = g_icolour #Need to read in g and i colour earlier from the table just as the other parameters like z are read in.
        
    #print( 'Processing', ID, ', file', ind, 'of', len(objID_in))
    
    #if galclass_exclude==2 and galclass!=1:
    
    if (galclass==2 )or (galclass==3 ) or ( galclass==1 and galclass_ex!=1) or (galclass==4 and galclass_ex!=1):
        print( 'Processing', ID, ', file', ind, 'of', len(objID_in))
    else:
        return
    
    map=glob.glob(fitdir+line.strip()+'_*1_comp.fits.gz')
    #print(map)
    #print(map[0])
    
    maps=fits.open(map[0])
    data=glob.glob(datadir+line.strip()+'_red*.fits*')
    
    data_r=fits.open(data[0])
    seeing=data_r[0].header['PSFFWHM']
    seeing_re=seeing/reff
    #print(data)
    #maps=fits.open(fitdir+str(ID)+'_1_comp.fits') #lzifu fits
    #data_r=fits.open(datadir+str(ID)+'_R.fits')   #SAMI data cube
    
    ###################################################################################
    ################################ calculate EW_ha total ############################
    ###################################################################################
    #cube  =data_r['PRIMARY'].data
    cube=data_r['BINNED_FLUX_ANNULAR'].data
    #cube=data_r['BINNED_FLUX_SECTORS'].data
    sumcube = np.nansum(np.nansum(cube,axis=1),axis=1)#add all pixel together
    #print(len(sumcube))
    wave_r=uf.wavelength(header=data_r[0].header)
    CDELT3=data_r[0].header['CDELT3']
    ha_ind1=uf.find_nearest_index(wave_r,uf.lines['HALPHA']*(1.0+z)) #find Halpha line in the spectrum
    #print(ha_ind1)
    ha_con =(sum(sumcube[ha_ind1-20:ha_ind1-10])/len(sumcube[ha_ind1-20:ha_ind1-10])+sum(sumcube[ha_ind1+10:ha_ind1+20])/len(sumcube[ha_ind1+10:ha_ind1+20]))/2 # calculate the continue flux
    #print(ha_con)
    ha_EW1=0
    for a in sumcube[ha_ind1-20:ha_ind1+20]:
        ha_EW1=ha_EW1+(1-a/ha_con)*CDELT3
    ha_EW1=abs(ha_EW1)
    ###################################################################################

    ha=maps['HALPHA'].data[1]
    hb=maps['HBETA'].data[1]
    ha_err=maps['HALPHA_ERR'].data[1]
    hb_err=maps['HBETA_ERR'].data[1]
    n2=maps['NII6583'].data[1]
    n2_err=maps['NII6583_ERR'].data[1]
    o3=maps['OIII5007'].data[1]
    o3_err=maps['OIII5007_ERR'].data[1]
    s26716=maps['SII6716'].data[1]
    s26731=maps['SII6731'].data[1]
    s2=s26716+s26731
    s2_err=maps['SII6716_ERR'].data[1]
    r_cont=maps['R_CONTINUUM'].data
    

    n2ha=np.log10(n2/ha)
    o3hb=np.log10(o3/hb)
    s2ha=np.log10(s2/ha)

    amask2=np.ones(n2ha.shape)
    #amask2[np.where((np.isnan(n2ha)==True))]=np.nan
    amask2[np.where((np.isnan(o3hb)==True)|(np.isnan(s2/ha)==True)|(np.isnan(n2ha)==True))]=np.nan
    #n2ha=n2ha*amask2
    #o3hb=o3hb*amask2
    
    n2ha_list=n2ha.flatten()
    x = [a for a in n2ha_list if (np.isnan(a)==False )]
    if len(x)<=5:
        print( len(x))
    
    outputtable.write ("{0:.0f}".format(objID) +" "+"{0:.2f}".format(reff) +" " + "{0:.2f}".format(pa_ori) +" "+"{0:.5f}".format(ellip)+" "+"{0:.5f}".format(z)+" "+ "{0:.4f}".format(gMINUSi) +" "+ "{0:.2f}".format(Mstar) +" "+ "{0:.0f}".format(BCGCATID) +" "+ "{0:.8E}".format(MassA))
    
    for i in range(9):
        if i == 0:
            #small
            n2ha[np.where((n2ha>0.0453))]=0.0453
            n2ha[np.where((n2ha<-0.402))]=-0.402
            ratio=(n2ha+0.402)/(0.0453+0.402)
        if i == 1:
            #middle
            n2ha[np.where((n2ha>0.11))]=0.11
            n2ha[np.where((n2ha<-0.438))]=-0.438
            ratio=(n2ha+0.438)/(0.11+0.438)
        if i == 2:
            #large
            n2ha[np.where((n2ha>0.199))]=0.199
            n2ha[np.where((n2ha<-0.457))]=-0.457
            ratio=(n2ha+0.457)/(0.199+0.457)
        if i == 3:
            #small threshold
            o3hb[np.where((o3hb>0.562))]=0.562
            o3hb[np.where((o3hb<-0.398))]=-0.398
            ratio=(o3hb+0.398)/(0.562+0.398)
        if i == 4:
            #middle
            o3hb[np.where((o3hb>0.7))]=0.7
            o3hb[np.where((o3hb<-0.473))]=-0.473
            ratio=(o3hb+0.473)/(0.473+0.7)
        if i == 5:
            #large
            o3hb[np.where((o3hb>0.891))]=0.891
            o3hb[np.where((o3hb<-0.516))]=-0.516
            ratio=(o3hb+0.516)/(0.516+0.891)
        if i == 6:
            #small
            s2ha[np.where((s2ha>-0.089))]=-0.089
            s2ha[np.where((s2ha<-0.328))]=-0.328
            ratio=(s2ha+0.328)/(-0.089+0.328)
        if i == 7:
            #middle
            s2ha[np.where((s2ha>-0.022))]=-0.022
            s2ha[np.where((s2ha<-0.362))]=-0.362
            ratio=(s2ha+0.362)/(-0.022+0.362)
        if i == 8:
            #large
            s2ha[np.where((s2ha>0.082))]=0.082
            s2ha[np.where((s2ha<-0.422))]=-0.422
            ratio=(s2ha+0.422)/(0.082+0.422)
            
        ha=ha*(1-ratio)
        n2hamask=np.ones(ha.shape)
        n2hamask[np.where( (1-ratio)==0 )]=np.nan
        ha=ha*n2hamask

        #Dust corrected ha values
        ha_dust_corr=ha*((ha/hb)/2.86)**2.36

        #Replace the dodgy values in this array. Where BD < 2.86 assume no dust correction. Also reject unphysically high values for BD and low S/N hbeta
        badpix=np.where((hb/hb_err<3) | (ha/hb < 2.86 ) )#| (ha/hb>30))
        ha_dust_corr[badpix]=ha[badpix]
        
        #Make a mask of the non-SF spaxels. 1=SF, 0=non-SF
        bptmask=uf.is_sf_array(n2,ha,o3,hb)

        d_bd=(2.86**-2.36)*np.sqrt((2.36*ha_err/ha)**2 + (2.36*hb_err/hb)**2)
        ha_dust_corr_err=ha_dust_corr*np.sqrt((ha_err/ha)**2 + d_bd**2)
        #Create an r_continuum image
        r_contmap=np.nan_to_num(scipy.nanmedian(r_cont,axis=0))
        #print(r_contmap)
        contmask=np.ones(r_contmap.shape)
        contmask[np.where(r_contmap<0)]=0
        #print(r_contmap)
        #cen=scipy.ndimage.center_of_mass(r_contmap)
        cen=(25,25)
        #Create an array that contains the distacnes
        distarr=uf.ellip_distarr(size=[50,50],centre=cen,pa=pa,ellip=ellip)
        aperture_distarr=uf.ellip_distarr(size=[50,50],centre=[25,25],pa=0,ellip=0)
        mask=np.ones(ha.shape)
        mask[np.where((np.isinf(ha_dust_corr)==True))]=0
        mask[np.where(aperture_distarr>14)]=0 #Masking exterior of the bundle
        #mask=mask*bptmask #apply the BPT mask to the pixel mask
        r,rp=uf.radial_profile(ha_dust_corr,distarr=distarr,mask=mask)
        #print(r,rp)
        mask=mask*contmask
        goodpix=np.where(np.isnan(rp)==False)
        r=r[goodpix]
        rp=rp[goodpix]
        r2,rp2=uf.radial_profile(ha_dust_corr_err,distarr=distarr,mask=mask)
        goodpix=np.where(np.isnan(rp2)==False)
        r2=r2[goodpix]
        rp2=rp2[goodpix]
        
        #Calculate equivalent width of Halpha
        wave_r=uf.wavelength(header=data_r[0].header)
        ha_ind=uf.find_nearest_index(wave_r,uf.lines['HALPHA']*(1.0+z))
        ha_cont=(r_cont[ha_ind-50,:,:]+r_cont[ha_ind+50,:,:])/2.0
        ha_EW=ha/ha_cont
        #print(ha_EW[25][25],ha[25][25],ha_cont[25][25])
        
        r3,rp3=uf.radial_profile(ha_EW,distarr=distarr,mask=mask)
        goodpix=np.where(np.isnan(rp3)==False)
        r3=r3[goodpix]
        rp3=rp3[goodpix]
        r4,rp4=uf.radial_profile(r_contmap,distarr=distarr,mask=contmask)
        goodpix=np.where((np.isnan(rp4)==False) & (rp4!=0))
        r4=r4[goodpix]
        rp4=rp4[goodpix]
        #Output the results into the arrays.
        #Maps
        ha_dc_map=ha_dust_corr
        cont_map=r_contmap
        ha_map=ha
        ha_ew_map=ha_EW

        #######################################################################################
        #########################      calculate SFR   ########################################
        #######################################################################################
        #WMAP7.luminosity_distance(z) unit of Mpc
        #dl=WMAP7.luminosity_distance(z).value*3.085677*10**24
        sumHafluxallspaxels = np.nansum(ha_dust_corr[np.logical_and(ha.astype(float)/ha_err.astype(float)>3.,hb.astype(float)/hb_err.astype(float)>3.)])
        #print('@'*50)
        #print(sumHafluxallspaxels)       #10^-16 ergs/cm^2/s=e10^-13 w/m^2
        dl=WMAP7.luminosity_distance(z).value*3.085677*10**22 #m
        #print('dl '*50,dl)
        Lha=sumHafluxallspaxels*(4*np.pi*(dl**2))/1e19   #10^-16 ergs/cm^2/s=10^-19 w/m^2 >> 10^-19 w
        #print('Lha '*50,Lha)
        SFR=(Lha/(2.16e34)*1.989e30)
        SFRsun=Lha/(2.16e34)
        #print(SFRsun)
        if SFRsun > 0:
            SFRsun=math.log10(SFRsun)
        else:
            SFRsun=0.0
        #print(SFRsun)
        #######################################################################################

        scrfluxes=np.zeros(100)*np.nan
        scrad=np.zeros(100)*np.nan
        
        #R50 measures
        
        a=np.ones(r_contmap.shape)
        b=np.ones(ha.shape)
        #print(a)
        
        try:
            #Curve of growth for Halpha (dust corrected)
            r5,cog_ha=uf.curve_of_growth(ha_dust_corr,distarr=distarr,mask=mask)
            #Curve of growth for continuum
            r6,cog_cont=uf.curve_of_growth(r_contmap,distarr=distarr,mask=contmask)
            #Curve of growth for #Halpha
            r7,cog_ha_ndc=uf.curve_of_growth(ha,distarr=distarr,mask=b)
            #Calculate r50 for these curves of growth
            #print(r5)
            ha_r50_ind=get_x(r5,cog_ha,0.5)
            #ha_ndc_r50_ind=get_x(r7,cog_ha_ndc,0.5)
            cont_r50_ind=get_x(r6,cog_cont,0.5)
            #ha_r80_ind=get_x(r5,cog_ha,0.8)
            #cont_r80_ind=get_x(r6,cog_cont,0.8)
            #ha_r20_ind=get_x(r5,cog_ha,0.2)
            #cont_r20_ind=get_x(r6,cog_cont,0.2)
        except:
            ha_r50_ind=np.nan
            #print(ha_r50_ind)
            cont_r50_ind=np.nan
            ha_ndc_r50_ind=np.nan
            cog_cont=np.nan
            cog_ha=np.nan
            ha_r80_ind=np.nan
            cont_r80_ind=np.nan
            ha_r20_ind=np.nan
            cont_r20_ind=np.nan
        
        ha_r50=ha_r50_ind
        #print( "ha_r50",ha_r50)

        cont_r50=cont_r50_ind
        #print( "cont_r50",cont_r50)
        concentration = ha_r50/cont_r50
        if concentration > 0:
            concentration = math.log10(concentration)
        else:
            concentration = -999
        #print(concentration)
        
        outputtable.write (" "+"{0:.8f}".format(concentration)+" "+"{0:.6f}".format(ha_EW1)+" "+"{0:.6f}".format(SFRsun)+" ") #This formats the values and writes them to your table.
    
    outputtable.write ("{0:.0f}".format(galclass) +" "+ "{0:.0f}".format(galclass_ex) +" "+ "{0:.6f}".format(seeing) +" "+ "{0:.7f}".format(seeing_re) +"\n")
    return


IDlist = open(filedir+'/list/gama/gamalist_havedata.txt','r')
lines = IDlist.readlines()[1:]
i=0
for line in lines:
    i=i+1
    print( line.strip(),' Process ',i,' of ',len(lines))
    map=glob.glob(fitdir+str(line.strip())+'_*1_comp.fits.gz')
    #print(map)
    #print(map[0])
    if map ==[]:
        continue

    data=glob.glob(datadir+str(line.strip())+'_*red*.fits*')
    #print(data)
    if data==[]:
        continue
    radial(np.int(line.strip()))

IDlist.close()

'''
#####################################################
#create ID list txt file
IDoutputtable = open(filedir+'/data/IDlist.txt','w')
files=glob.glob(filedir+'/data/data_r/*.gz')
for file in files:
    header =fits.getheader(file)
    objname=header['NAME']
    IDoutputtable.write(str(objname)+'\n')
    
IDoutputtable.close()

'''


outputtable.close()  #to close the table file



##########################################################
###### Multi_processing (Optional)

#Number of CPUs to use
#ncpu=2

#Initiate the CPU pool
#p=Pool(ncpu)

#names=[250277,609396]

#Begin the calculation
#_=p.map(radial,names,chunksize=2)
