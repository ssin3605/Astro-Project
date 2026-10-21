import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
#import statsmodels.api as sm
import scipy as sp
#from scipy.stats import norm
#from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn import linear_model
#from sklearn.pipeline import Pipeline
#import statsmodels.api as sm
import math
import matplotlib
from matplotlib import gridspec
from astropy.cosmology import WMAP9 as cosmo
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
AutoMinorLocator)
from scipy import stats

#outputtable = open('/Users/ice/Desktop/SFH_Di_CATID.txt','w')   #Create a file to write to
#outputtable.write ("#CATID \n")

data = np.loadtxt('/Users/ice/Dropbox (Sydney Uni)/PHDprocess/dncode/dataAdam/gama_v0.12/gamaout_sf_hbsn3_v0.12.txt', dtype='float',comments='#')
data_c =  np.loadtxt('/Users/ice/Dropbox (Sydney Uni)/PHDprocess/dncode/dataAdam/gama_v0.12/gamaout_com_v0.12.txt', dtype='float',comments='#')

clu_data = np.loadtxt('/Users/ice/Dropbox (Sydney Uni)/PHDprocess/dncode/dataAdam/cluster_v0.12/clusterout_sf_hbsn3_v0.12.txt', dtype='float',comments='#')
clu_data_c = np.loadtxt('/Users/ice/Dropbox (Sydney Uni)/PHDprocess/dncode/dataAdam/cluster_v0.12/cluout_com_v0.12.txt', dtype='float',comments='#')
#####################################################

conc_f=[] ; mass_f=[]; sfr_f=[]; g_icolor=[]; radius=[]; size=[]; redshift=[]
conc_l=[] ; mass_l=[]; sfr_l=[]; conc_l_BCG=[] ; mass_l_BCG=[]
conc_h=[] ; mass_h=[]; sfr_h=[]; conc_h_BCG=[] ; mass_h_BCG=[]

for CATID ,  r_e, PA, ellip, z,  g_i, Mstar,BCGCATID, MassA,  CIndex, totalHagas, ha_r50, cont_r50,      ha_EW, SFR, galclass, galclass_ex, seeing, seeing_r_e  in data:
    #sf
    #if galclass==0:
    #if  ((galclass==1 and galclass_ex==1)or (galclass==4 and galclass_ex==1))and   0<r_e <15 and ha_EW > 1 and 0<ellip <0.7 and seeing_r_e < 0.75 and seeing < 4 :
    if  ((galclass==1 and galclass_ex==1))and MassA ==0. and BCGCATID==0 and   0<r_e <15 and ha_EW > 1 and 0<ellip <0.7 and seeing_r_e < 0.75 and seeing < 4 and SFR!=0 and SFR-Mstar >-11.25:
        #print(CATID)
        conc_f.append(CIndex)
        sfr_f.append(SFR-Mstar)
        mass_f.append(Mstar)
        #outputtable.write(str(CATID)+ "\n")
    if ((galclass==1 and galclass_ex==1)) and 0. <MassA < 10**12.5   and 0 <r_e <15 and ha_EW >1 and 0<ellip <0.7 and seeing_r_e < 0.75 and seeing < 4 and SFR!=0 and SFR-Mstar >-11.25:
        conc_l.append(CIndex)
        sfr_l.append(SFR-Mstar)
        mass_l.append(Mstar)
        #outputtable.write(str(CATID)+ "\n")
        if CATID==BCGCATID:
            conc_l_BCG.append(CIndex)
            mass_l_BCG.append(Mstar)
    if  ((galclass==1 and galclass_ex==1)) and 10**14 >MassA >10**12.5  and 0<r_e <15 and ha_EW >1 and 0<ellip <0.7 and seeing <4 and seeing_r_e <0.75 and SFR!=0 and SFR-Mstar >-11.25:
        conc_h.append(CIndex)
        sfr_h.append(SFR-Mstar)
        mass_h.append(Mstar)
        if CIndex < -0.2:
            print(CATID)
        #outputtable.write(str(CATID)+ "\n")
        if CATID==BCGCATID:
            conc_h_BCG.append(CIndex)
            mass_h_BCG.append(Mstar)
        #print(cosmo.kpc_proper_per_arcmin(z).value/60*15)
        #print(int(CATID))
        size.append(cosmo.kpc_proper_per_arcmin(z).value/60*r_e)
        redshift.append(z)
        g_icolor.append(g_i)
        radius.append(r_e)
        #outputtable.write ("{0:.0f}".format(CATID)+"\n")
    
conc_f_2=[] ; mass_f_2=[]; sfr_f_2=[]; err_neg_f_2=[];err_pos_f_2=[]
conc_l_2=[] ; mass_l_2=[]; sfr_l_2=[]; err_neg_l_2=[];err_pos_l_2=[]
conc_h_2=[] ; mass_h_2=[]; sfr_h_2=[]; err_neg_h_2=[];err_pos_h_2=[]


for CATID , r_e, PA, ellip, z,  g_i, Mstar, BCGCATID, MassA, CIndex_n2ha_s , ha_EW_n_s, SFR_n2ha_s, CIndex_n2ha_m , ha_EW_n_m, SFR_n2ha ,     CIndex_n2ha_l  ,ha_EW_n_l, SFR_n2ha_l, CIndex_o3hb_s ,ha_EW_o_s ,SFR_o3hb_s ,  CIndex_o3hb_m  ,ha_EW_o_m, SFR_o3hb , CIndex_o3hb_l  , ha_EW_o_l, SFR_o3hb_l,   CIndex_s2ha_s ,ha_EW_s_s, SFR_s2ha_s , CIndex_s2ha_m , ha_EW_s_m, SFR_s2ha   ,   CIndex_s2ha_l ,ha_EW_s_l , SFR_s2ha_l  , galclass, galclass_ex, seeing, seeing_r_e  in data_c:

    if  ((galclass==1 and galclass_ex==2 )or(galclass==2 and galclass_ex==1))  and MassA ==0 and BCGCATID==0 and 0<r_e <15 and ha_EW_n_m >1 and 0<ellip <0.7 and seeing_r_e < 0.75 and seeing < 4 and SFR_n2ha!=0 and (SFR_n2ha-Mstar)>-11.25:
        conc_f_2.append(CIndex_n2ha_m)
        sfr_f_2.append(SFR_n2ha-Mstar)
        mass_f_2.append(Mstar)
        err_neg_f_2.append(abs(CIndex_n2ha_m-CIndex_n2ha_s))
        err_pos_f_2.append(abs(CIndex_n2ha_m-CIndex_n2ha_l))
        #outputtable.write(str(CATID)+ "\n")
    
    if  ((galclass==1 and galclass_ex==2 )or (galclass==2 and galclass_ex==1) )  and 0<MassA < 10**12.5 and 0<r_e <15 and ha_EW_n_m >1 and 0<ellip <0.7 and seeing <4 and seeing_r_e <0.75  and (SFR_n2ha-Mstar) >-11.25 and SFR_n2ha!=0 and (SFR_n2ha-Mstar)>-11.25:
        conc_l_2.append(CIndex_n2ha_m)
        sfr_l_2.append(SFR_n2ha-Mstar)
        mass_l_2.append(Mstar)
        err_neg_l_2.append(abs(CIndex_n2ha_m-CIndex_n2ha_s))
        err_pos_l_2.append(abs(CIndex_n2ha_m-CIndex_n2ha_l))
        #outputtable.write(str(CATID)+ "\n")
        if CATID==BCGCATID:
            conc_l_BCG.append(CIndex_n2ha_m)
            mass_l_BCG.append(Mstar)
    if  ((galclass==1 and galclass_ex==2 )or (galclass==2 and galclass_ex==1) ) and 10**14>MassA > 10**12.5 and 0<r_e <15 and 0<ellip <0.7 and seeing <4 and seeing_r_e <0.75 and ha_EW_o_m >1  and (SFR_n2ha-Mstar) >-11.25 and SFR_n2ha!=0 and (SFR_n2ha-Mstar)>-11.25:# and Mstar>10.5 :
        conc_h_2.append(CIndex_n2ha_m)
        sfr_h_2.append(SFR_n2ha-Mstar)
        mass_h_2.append(Mstar)
        err_neg_h_2.append(abs(CIndex_n2ha_m-CIndex_n2ha_s))
        err_pos_h_2.append(abs(CIndex_n2ha_m-CIndex_n2ha_l))
        #outputtable.write(str(CATID)+ "\n")
        if CATID==BCGCATID:
            conc_h_BCG.append(CIndex_n2ha_m)
            mass_h_BCG.append(Mstar)
    
Redshift=np.array(redshift)
Size=np.array(size);G_icolor=np.array(g_icolor);Radius=np.array(radius)
Conc_f=np.array(conc_f);Logmstar_f=np.array(mass_f);Sfr_f=np.array(sfr_f)
Conc_l=np.array(conc_l);Logmstar_l=np.array(mass_l);Sfr_l=np.array(sfr_l);Conc_l_BCG=np.array(conc_l_BCG);Logmstar_l_BCG=np.array(mass_l_BCG)
Conc_h=np.array(conc_h);Logmstar_h=np.array(mass_h);Sfr_h=np.array(sfr_h);Conc_h_BCG=np.array(conc_h_BCG);Logmstar_h_BCG=np.array(mass_h_BCG)
print('SF',len(Conc_f),len(Conc_l),len(Conc_h))

Conc_f_2=np.array(conc_f_2);Logmstar_f_2=np.array(mass_f_2);Sfr_f_2=np.array(sfr_f_2) ;Err_neg_f_2=np.array(err_neg_f_2);Err_pos_f_2=np.array(err_pos_f_2)
Conc_l_2=np.array(conc_l_2);Logmstar_l_2=np.array(mass_l_2);Sfr_l_2=np.array(sfr_l_2);Err_neg_l_2=np.array(err_neg_l_2);Err_pos_l_2=np.array(err_pos_l_2)
Conc_h_2=np.array(conc_h_2);Logmstar_h_2=np.array(mass_h_2);Sfr_h_2=np.array(sfr_h_2);Err_neg_h_2=np.array(err_neg_h_2);Err_pos_h_2=np.array(err_pos_h_2)
print('SF_2',len(Conc_f_2),len(Conc_l_2),len(Conc_h_2))

conc_f_com=[] ; mass_f_com=[]; sfr_f_com=[];err_neg_f=[];err_pos_f=[]
conc_l_com=[] ; mass_l_com=[]; sfr_l_com=[];err_neg_l=[];err_pos_l=[]
conc_h_com=[] ; mass_h_com=[]; sfr_h_com=[];err_neg_h=[];err_pos_h=[]
for CATID , r_e, PA, ellip, z,  g_i, Mstar, BCGCATID, MassA,  CIndex_n2ha_s , ha_EW_n_s, SFR_n2ha_s, CIndex_n2ha_m , ha_EW_n_m, SFR_n2ha ,     CIndex_n2ha_l  ,ha_EW_n_l, SFR_n2ha_l, CIndex_o3hb_s ,ha_EW_o_s ,SFR_o3hb_s ,  CIndex_o3hb_m  ,ha_EW_o_m, SFR_o3hb , CIndex_o3hb_l  , ha_EW_o_l, SFR_o3hb_l,   CIndex_s2ha_s ,ha_EW_s_s, SFR_s2ha_s , CIndex_s2ha_m , ha_EW_s_m, SFR_s2ha   ,   CIndex_s2ha_l ,ha_EW_s_l , SFR_s2ha_l , galclass, galclass_ex, seeing, seeing_r_e  in data_c:
    #if galclass==0:
    #if  ((galclass_ex==1 and galclass!=1) or galclass_ex==2 or galclass_ex==3 or (galclass_ex==4 and galclass!=1))  and MassA ==0 and BCGCATID==0 and 0<r_e <15 and ha_EW_n_m >1 and 0<ellip <0.7 and seeing_r_e < 0.75 and seeing < 4 and -0.5 < CIndex_n2ha_m < 0.25:
    if  ((galclass==1 and galclass_ex==3 )or (galclass==2 and galclass_ex!=1) or galclass==3 )  and MassA ==0 and BCGCATID==0 and 0<r_e <15 and ha_EW_n_m >1 and 0<ellip <0.7 and seeing_r_e < 0.75 and seeing < 4  and (SFR_n2ha-Mstar)>-11.25 and SFR_n2ha!=0:
        conc_f_com.append(CIndex_n2ha_m)
        err_neg_f.append(abs(CIndex_n2ha_m-CIndex_n2ha_s))
        err_pos_f.append(abs(CIndex_n2ha_m-CIndex_n2ha_l))
        #print(err_neg_f)
        sfr_f_com.append(SFR_n2ha-Mstar)
        mass_f_com.append(Mstar)
        #outputtable.write(str(CATID)+ "\n")
    
    if  ((galclass==1 and galclass_ex==3 )or (galclass==2 and galclass_ex!=1) or galclass==3 )  and 0<MassA < 10**12.5 and 0<r_e <15 and ha_EW_n_m >1 and 0<ellip <0.7 and seeing <4 and seeing_r_e <0.75  and (SFR_n2ha-Mstar)>-11.25 and SFR_n2ha!=0:
        #print(CATID)
        conc_l_com.append(CIndex_n2ha_m)
        err_neg_l.append(abs(CIndex_n2ha_m-CIndex_n2ha_s))
        err_pos_l.append(abs(CIndex_n2ha_m-CIndex_n2ha_l))
        sfr_l_com.append(SFR_n2ha-Mstar)
        #outputtable.write(str(CATID)+ "\n")
        mass_l_com.append(Mstar)
    if  ((galclass==1 and galclass_ex==3 )or (galclass==2 and galclass_ex!=1) or galclass==3 ) and 10**14 >MassA > 10**12.5 and 0<r_e <15 and 0<ellip <0.7 and seeing <4 and seeing_r_e <0.75 and ha_EW_n_m >1 and (SFR_n2ha-Mstar)>-11.25 and SFR_n2ha!=0:# and Mstar>10.5 :
        #print(CATID)
        conc_h_com.append(CIndex_n2ha_m)
        err_neg_h.append(abs(CIndex_n2ha_m-CIndex_n2ha_s))
        err_pos_h.append(abs(CIndex_n2ha_m-CIndex_n2ha_l))
        sfr_h_com.append(SFR_n2ha-Mstar)
        mass_h_com.append(Mstar)
        #outputtable.write(str(CATID)+ "\n")
    #if 0<r_e <15 and 0<ellip <0.7 and seeing <4 and seeing_r_e <0.75 and ha_EW_n_m >1:
        '''
        conc.append(CIndex_n2ha_m)
        err_neg.append(CIndex_n2ha_m-CIndex_n2ha_l)
        err_pos.append(CIndex_n2ha_s- CIndex_n2ha_m)
        sfr.append(SFR_n2ha-Mstar)
        
        conc.append(CIndex_o3hb_m)
        err_neg.append(CIndex_o3hb_m-CIndex_o3hb_l)
        err_pos.append(CIndex_o3hb_s-CIndex_o3hb_m)
        sfr.append(SFR_o3hb-Mstar)
        
        conc.append(CIndex_s2ha_m)
        err_neg.append(CIndex_s2ha_m-CIndex_s2ha_l)
        err_pos.append(CIndex_s2ha_s- CIndex_s2ha_m)
        sfr.append(SFR_s2ha-Mstar)
        '''
Conc_f_com=np.array(conc_f_com);Logmstar_f_com=np.array(mass_f_com);Sfr_f_com=np.array(sfr_f_com);Err_neg_f=np.array(err_neg_f);Err_pos_f=np.array(err_pos_f)
Conc_l_com=np.array(conc_l_com);Logmstar_l_com=np.array(mass_l_com);Sfr_l_com=np.array(sfr_l_com);Err_neg_l=np.array(err_neg_l);Err_pos_l=np.array(err_pos_l)
Conc_h_com=np.array(conc_h_com);Logmstar_h_com=np.array(mass_h_com);Sfr_h_com=np.array(sfr_h_com);Err_neg_h=np.array(err_neg_h);Err_pos_h=np.array(err_pos_h)
print('Com',len(Conc_f_com),len(Conc_l_com),len(Conc_h_com))



conc_c=[] ; mass_c=[]; sfr_c=[];
for CATID ,  r_e, PA, ellip, z,  g_i, Mstar,  CIndex, totalHagas, ha_r50, cont_r50,      ha_EW, SFR, galclass, galclass_ex, seeing, seeing_r_e  in clu_data:
    
    if   ((galclass==1 and galclass_ex==1)) and   0<r_e <15 and ha_EW > 1 and 0<ellip <0.7 and seeing_r_e < 0.75 and seeing < 4  and SFR!=0 and (SFR-Mstar) > -11.25 and CIndex<0.2:#and ( CIndex>0.05 or CIndex<-0.3):
        #print(int(CATID))#,CIndex)
        conc_c.append(CIndex)
        sfr_c.append(SFR-Mstar)
        mass_c.append(Mstar)
        
conc_c_2=[] ; mass_c_2=[]; sfr_c_2=[]; err_neg_c_2=[];err_pos_c_2=[]
for CATID , r_e, PA, ellip, z,  g_i, Mstar,  CIndex_n2ha_s , ha_EW_n_s, SFR_n2ha_s, CIndex_n2ha_m , ha_EW_n_m, SFR_n2ha ,     CIndex_n2ha_l  ,ha_EW_n_l, SFR_n2ha_l, CIndex_o3hb_s ,ha_EW_o_s ,SFR_o3hb_s ,  CIndex_o3hb_m  ,ha_EW_o_m, SFR_o3hb , CIndex_o3hb_l  , ha_EW_o_l, SFR_o3hb_l,   CIndex_s2ha_s ,ha_EW_s_s, SFR_s2ha_s , CIndex_s2ha_m , ha_EW_s_m, SFR_s2ha   ,   CIndex_s2ha_l ,ha_EW_s_l , SFR_s2ha_l , galclass, galclass_ex, seeing, seeing_r_e  in clu_data_c:
    #if galclass_ex==0:
    if ((galclass==1 and galclass_ex==2 ) or (galclass==2 and galclass_ex==1) )and 0< r_e <15 and ha_EW_n_m > 1 and 0<ellip <0.7 and seeing_r_e < 0.75 and seeing < 4  and (SFR_n2ha-Mstar) >-11.25 and SFR_n2ha!=0:#  and ( CIndex>0.05 or CIndex<-0.3):
        #print(int(CATID))#,CIndex_n2ha_m)
        conc_c_2.append(CIndex_n2ha_m)
        sfr_c_2.append(SFR_n2ha-Mstar)
        mass_c_2.append(Mstar)
        err_neg_c_2.append(abs(CIndex_n2ha_m-CIndex_n2ha_s))
        err_pos_c_2.append(abs(CIndex_n2ha_m-CIndex_n2ha_l))
Conc_c=np.array(conc_c);Logmstar_c=np.array(mass_c);Sfr_c=np.array(sfr_c)
Conc_c_2=np.array(conc_c_2);Logmstar_c_2=np.array(mass_c_2);Sfr_c_2=np.array(sfr_c_2) ;Err_neg_c_2=np.array(err_neg_c_2);Err_pos_c_2=np.array(err_pos_c_2)

print('clusf',len(Conc_c))
print('clusf_2',len(Conc_c_2))

conc_c_com=[] ; mass_c_com=[]; sfr_c_com=[];err_neg_c=[];err_pos_c=[]
for CATID , r_e, PA, ellip, z,  g_i, Mstar,  CIndex_n2ha_s , ha_EW_n_s, SFR_n2ha_s, CIndex_n2ha_m , ha_EW_n_m, SFR_n2ha ,     CIndex_n2ha_l  ,ha_EW_n_l, SFR_n2ha_l, CIndex_o3hb_s ,ha_EW_o_s ,SFR_o3hb_s ,  CIndex_o3hb_m  ,ha_EW_o_m, SFR_o3hb , CIndex_o3hb_l  , ha_EW_o_l, SFR_o3hb_l,   CIndex_s2ha_s ,ha_EW_s_s, SFR_s2ha_s , CIndex_s2ha_m , ha_EW_s_m, SFR_s2ha   ,   CIndex_s2ha_l ,ha_EW_s_l , SFR_s2ha_l  , galclass, galclass_ex, seeing, seeing_r_e  in clu_data_c:
    #if galclass_ex==0:
    if  ((galclass==1 and galclass_ex==3 )or (galclass==2 and galclass_ex!=1) or galclass==3 ) and 0< r_e <15 and ha_EW_n_m > 1 and 0<ellip <0.7 and seeing_r_e < 0.75 and seeing < 4 and -0.5<CIndex_n2ha_m<0.25   and (SFR_n2ha-Mstar) >-11.25 and SFR_n2ha!=0 :
        #print(int(CATID))
        conc_c_com.append(CIndex_n2ha_m)
        err_neg_c.append(abs(CIndex_n2ha_m-CIndex_n2ha_s))
        err_pos_c.append(abs(CIndex_n2ha_m-CIndex_n2ha_l))
        sfr_c_com.append(SFR_n2ha-Mstar)
        mass_c_com.append(Mstar)
Conc_c_com=np.array(conc_c_com);Logmstar_c_com=np.array(mass_c_com);Sfr_c_com=np.array(sfr_c_com);Err_neg_c=np.array(err_neg_c);Err_pos_c=np.array(err_pos_c)
print('Clucom',len(Conc_c_com))

#outputtable.close()
#plt.figure(figsize=(6, 6))
cm = plt.cm.get_cmap('coolwarm_r')

cccf=conc_f_com+conc_f
cccl=conc_l_com+conc_l
ccch=conc_h_com+conc_h
cccc=conc_c_com+conc_c
len(np.where(np.array(cccf)>=-0.2)[0])
len(np.where(np.array(cccf)<-0.2)[0])
from scipy.stats import binom


n=len(cccf)
p=len(np.where(np.array(cccf)<-0.2)[0])/n
print(p, np.sqrt(p*(1-p)/n))
binom.stats(n,p)

n=334;p=0.0838
r_values = list(range(n + 1))
dist = [binom.pmf(r, n, p) for r in r_values ]
#plt.bar(r_values, dist)
#plt.show()
#ccc=conc_c_com+conc_l_com+conc_h_com+conc_f_com




matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

tickfontsize=5
labelsize=14

'''
fig, ax = plt.subplots(1,4,figsize=(12, 3.5),
                       sharex='all',
                       sharey='row')
plt.subplots_adjust(left = 0.1 ,right = 0.9,bottom = 0.15,top = 0.9 ,wspace = 0,hspace = 0)

#fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.9, 0.15, 0.01, 0.75])
fig.text(0.5, 0.04, 'log'+r'$(M_{*}/M_{\odot})$', ha='center',fontsize=labelsize)
fig.text(0.03, 0.5, r'$log_{10}(r_{50,H\alpha}/r_{50,cont})$', va='center', rotation='vertical',fontsize=(labelsize+3))

#ax[0,0].set_yticks([-0.45, -0.3, -0.15, 0,0.15])
#ax[0,0].set_yticklabels( [-0.45, -0.3, -0.15, 0,0.15],fontsize=tickfontsize)
x=(7,12);y=(0,0);dotsize=13

ax[0].set_ylim(-0.65,0.4)
ax[0].set_xticks([8,9,10,11])
ax[0].xaxis.set_minor_locator(AutoMinorLocator(5))
ax[0].yaxis.set_minor_locator(AutoMinorLocator(3))
#ax[0,0].set_xticklabels( [8,9,10,11],fontsize=tickfontsize)
ax[0].set_xlim(7.5,11.5)

im=ax[0].scatter(Logmstar_f,Conc_f,c=Sfr_f, cmap=cm,vmin=-11,vmax=-9.5,s=dotsize,marker='o')
ax[0].plot(x,y,linewidth=1,color='darkgrey',linestyle='--')
ax[0].set_title('Ungrouped',fontsize=labelsize)
ax[1].scatter(Logmstar_l,Conc_l,c=Sfr_l, cmap=cm,vmin=-11,vmax=-9.5,s=dotsize,marker='o')
ax[1].plot(x,y,linewidth=1,color='darkgrey',linestyle='--')
#ax[0,1].scatter(Logmstar_l_BCG,Conc_l_BCG,facecolors='none', edgecolors='black',s=35,marker='o',linewidth=0.5,label='BCG')
#ax[0,1].legend(loc='lower left',fontsize=10)
ax[1].set_title(r'$M_{G}<10^{12.5}M_{\odot}$',fontsize=labelsize)
ax[2].scatter(Logmstar_h,Conc_h,c=Sfr_h, cmap=cm,vmin=-11,vmax=-9.5,s=dotsize,marker='o')
ax[2].plot(x,y,linewidth=1,color='darkgrey',linestyle='--')
#ax[0,2].scatter(Logmstar_h_BCG,Conc_h_BCG,facecolors='none', edgecolors='black',s=35,marker='o',linewidth=0.5)
ax[2].set_title(r'$M_{G}>10^{12.5}M_{\odot}$',fontsize=labelsize)
ax[3].scatter(Logmstar_c,Conc_c,c=Sfr_c, cmap=cm,vmin=-11,vmax=-9.5,s=dotsize,marker='o')
ax[3].plot(x,y,linewidth=1,color='darkgrey',linestyle='--')
ax[3].set_title('Cluster',fontsize=labelsize)

cbar=fig.colorbar(im, cax=cbar_ax)
cbar.set_label(r'$\log(sSFR/yr^{-1}$)',fontsize=labelsize)

#plt.savefig('/Users/ice/Dropbox (Sydney Uni)/PHDprocess/dncode/plotAdam/paperimage/CIndexvsMass_withoutBCG_sf.pdf')

'''
fig, ax = plt.subplots(2,4,figsize=(12, 6),
                       sharex='all',
                       sharey='row')
plt.subplots_adjust(left = 0.1 ,right = 0.9,bottom = 0.1,top = 0.9 ,wspace = 0,hspace = 0)

#fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.9, 0.1, 0.01, 0.8])
fig.text(0.5, 0.04,r'$\log\mathrm{(M_{\star}/M_{\odot})}$', ha='center',fontsize=labelsize)
fig.text(0.04, 0.55, r'$\log_{10}\mathrm{(r_{50,H\alpha}/r_{50,cont})}$', va='center', rotation='vertical',fontsize=(labelsize+3))

#ax[0,0].set_yticks([-0.5, -0.4, -0.3, -0.0,0.15])
#ax[0,0].set_yticklabels( [-0.45, -0.3, -0.15, 0,0.15],fontsize=tickfontsize)
ax[0,0].set_ylim(-0.65,0.3)
ax[0,0].set_xticks([8,9,10,11])
ax[0,0].xaxis.set_minor_locator(AutoMinorLocator(5))
ax[0,0].yaxis.set_minor_locator(AutoMinorLocator(3))
#ax[0,0].set_xticklabels( [8,9,10,11],fontsize=tickfontsize)
ax[0,0].set_xlim(7.3,11.5)
#ax.set_yticks([-0.15, -0.1, 0, 0.1, 0.15])
ax[1,0].set_ylim(-0.65,0.3)
#ax[1,0].set_yticks([-0.45, -0.3, -0.15, 0,0.15,0.3])
ax[1,0].yaxis.set_minor_locator(AutoMinorLocator(3))

x=(7,12);y=(0,0)

im=ax[0,0].scatter(Logmstar_f,Conc_f,c=Sfr_f, cmap=cm,vmin=-11,vmax=-9.5,s=9,marker='o')
ax[0,0].scatter(Logmstar_f_2,Conc_f_2,c=Sfr_f_2, cmap=cm,vmin=-11,vmax=-9.5,s=9,marker='o')
ax[0,0].errorbar(Logmstar_f_2,Conc_f_2, (Err_neg_f_2,Err_pos_f_2), fmt='none',ecolor='black', elinewidth=1.5,alpha=0.5,label='central_SF')
ax[1,0].scatter(Logmstar_f_2,Conc_f_2,c=Sfr_f_2, cmap=cm,vmin=-11,vmax=-9.5,s=9,marker='o')
ax[1,0].errorbar(Logmstar_f_2,Conc_f_2, (Err_neg_f_2,Err_pos_f_2), fmt='none',ecolor='black', elinewidth=1.5,alpha=0.5)
ax[0,0].legend(loc='lower right',fontsize=10)
ax[0,0].plot(x,y,linewidth=1,color='darkgrey',linestyle='--')

ax[0,0].text(7.7, -0.58, '(a)', fontsize=12)
ax[0,1].scatter(Logmstar_l,Conc_l,c=Sfr_l, cmap=cm,vmin=-11,vmax=-9.5,s=9,marker='o')
ax[0,1].plot(x,y,linewidth=1,color='darkgrey',linestyle='--')
ax[0,1].scatter(Logmstar_l_2,Conc_l_2,c=Sfr_l_2, cmap=cm,vmin=-11,vmax=-9.5,s=9,marker='o')
ax[0,1].errorbar(Logmstar_l_2,Conc_l_2, (Err_neg_l_2,Err_pos_l_2), fmt='none',ecolor='black', elinewidth=1.5,alpha=0.5)
ax[1,1].scatter(Logmstar_l_2,Conc_l_2,c=Sfr_l_2, cmap=cm,vmin=-11,vmax=-9.5,s=9,marker='o')
ax[1,1].errorbar(Logmstar_l_2,Conc_l_2, (Err_neg_l_2,Err_pos_l_2), fmt='none',ecolor='black', elinewidth=1.5,alpha=0.5)
#ax[0,1].scatter(Logmstar_l_BCG,Conc_l_BCG,facecolors='none', edgecolors='black',s=35,marker='o',linewidth=0.5,label='BCG')
#ax[0,1].legend(loc='lower right',fontsize=10)
#ax[0,1].set_title(r'$M_{G}<10^{12.5}M_{\odot}$',fontsize=labelsize)

ax[0,1].text(7.7, -0.58, '(b)', fontsize=12)
ax[0,2].scatter(Logmstar_h,Conc_h,c=Sfr_h, cmap=cm,vmin=-11,vmax=-9.5,s=9,marker='o')
ax[0,2].plot(x,y,linewidth=1,color='darkgrey',linestyle='--')
ax[0,2].scatter(Logmstar_h_2,Conc_h_2,c=Sfr_h_2, cmap=cm,vmin=-11,vmax=-9.5,s=9,marker='o')
ax[0,2].errorbar(Logmstar_h_2,Conc_h_2, (Err_neg_h_2,Err_pos_h_2), fmt='none',ecolor='black', elinewidth=1.5,alpha=0.5)
ax[1,2].scatter(Logmstar_h_2,Conc_h_2,c=Sfr_h_2, cmap=cm,vmin=-11,vmax=-9.5,s=9,marker='o')
ax[1,2].errorbar(Logmstar_h_2,Conc_h_2, (Err_neg_h_2,Err_pos_h_2), fmt='none',ecolor='black', elinewidth=1.5,alpha=0.5)
#ax[0,2].scatter(Logmstar_h_BCG,Conc_h_BCG,facecolors='none', edgecolors='black',s=35,marker='o',linewidth=0.5)
#ax[0,2].set_title(r'$M_{G}>10^{12.5}M_{\odot}$',fontsize=labelsize)


ax[0,2].text(7.7, -0.58, '(c)', fontsize=12)
ax[0,3].scatter(Logmstar_c,Conc_c,c=Sfr_c, cmap=cm,vmin=-11,vmax=-9.5,s=9,marker='o')
ax[0,3].scatter(Logmstar_c_2,Conc_c_2,c=Sfr_c_2, cmap=cm,vmin=-11,vmax=-9.5,s=9,marker='o')
ax[0,3].errorbar(Logmstar_c_2,Conc_c_2, (Err_neg_c_2,Err_pos_c_2), fmt='none',ecolor='black', elinewidth=1.5,alpha=0.5)
ax[1,3].scatter(Logmstar_c_2,Conc_c_2,c=Sfr_c_2, cmap=cm,vmin=-11,vmax=-9.5,s=9,marker='o')
ax[1,3].errorbar(Logmstar_c_2,Conc_c_2, (Err_neg_c_2,Err_pos_c_2), fmt='none',ecolor='black', elinewidth=1.5,alpha=0.5)
ax[0,3].plot(x,y,linewidth=1,color='darkgrey',linestyle='--')
#ax[0,3].set_title('Cluster',fontsize=labelsize)

ax[0,0].set_title('Ungrouped',fontsize=labelsize)
ax[0,1].set_title(r'$\mathrm{M_{200} \leq 10^{12.5}M_{\odot}}$',fontsize=labelsize)
ax[0,2].set_title(r'$\mathrm{10^{12.5}M_{\odot}<M_{200} \leq 10^{14}M_{\odot}}$',fontsize=labelsize)
ax[0,3].set_title(r'$\mathrm{M_{200} > 10^{14}M_{\odot}}$',fontsize=labelsize)


ax[0,3].text(7.7, -0.58, '(d)', fontsize=12)

cbar=fig.colorbar(im, cax=cbar_ax)
cbar.set_label(r'$\log\mathrm{(sSFR/yr^{-1})}$',fontsize=labelsize)

ax[1,0].scatter(Logmstar_f,Conc_f,c=Sfr_f, cmap=cm,vmin=-11,vmax=-9.5,s=9,marker='o')
ax[1,0].plot(x,y,linewidth=1,color='darkgrey',linestyle='--')
ax[1,0].scatter(Logmstar_f_com,Conc_f_com,c=sfr_f_com, cmap=cm,vmin=-11,vmax=-9.5,s=12,marker='o')
ax[1,0].scatter(Logmstar_f_com,Conc_f_com,facecolors='none', edgecolors='r',s=35,marker='o',linewidth=0.5,label='central_LINER/AGN',alpha=0.5)
ax[1,0].errorbar(Logmstar_f_com,Conc_f_com, (Err_neg_f,Err_pos_f), fmt='none',ecolor='black', elinewidth=1.5,alpha=0.5)
ax[1,0].legend(loc='lower right',fontsize=10)
ax[1,0].text(7.7, -0.58, '(e)', fontsize=12)
ax[1,1].scatter(Logmstar_l,Conc_l,c=Sfr_l, cmap=cm,vmin=-11,vmax=-9.5,s=9,marker='o')
ax[1,1].plot(x,y,linewidth=1,color='darkgrey',linestyle='--')
ax[1,1].scatter(Logmstar_l_com,Conc_l_com,c=sfr_l_com, cmap=cm,vmin=-11,vmax=-9.5,s=12,marker='o')
ax[1,1].scatter(Logmstar_l_com,Conc_l_com,facecolors='none', edgecolors='r',s=35,marker='o',linewidth=0.5,alpha=0.5)
ax[1,1].errorbar(Logmstar_l_com,Conc_l_com, (Err_neg_l,Err_pos_l), fmt='none',ecolor='black', elinewidth=1.5,alpha=0.5)
ax[1,1].text(7.7, -0.58, '(f)', fontsize=12)
ax[1,2].scatter(Logmstar_h,Conc_h,c=Sfr_h, cmap=cm,vmin=-11,vmax=-9.5,s=9,marker='o')
ax[1,2].plot(x,y,linewidth=1,color='darkgrey',linestyle='--')
ax[1,2].scatter(Logmstar_h_com,Conc_h_com,c=sfr_h_com, cmap=cm,vmin=-11,vmax=-9.5,s=12,marker='o')
ax[1,2].scatter(Logmstar_h_com,Conc_h_com,facecolors='none', edgecolors='r',s=35,marker='o',linewidth=0.5,alpha=0.5)
ax[1,2].errorbar(Logmstar_h_com,Conc_h_com, (Err_neg_h,Err_pos_h), fmt='none',ecolor='black', elinewidth=1.5,alpha=0.5)
ax[1,2].text(7.7, -0.58, '(g)', fontsize=12)
ax[1,3].scatter(Logmstar_c,Conc_c,c=Sfr_c, cmap=cm,vmin=-11,vmax=-9.5,s=9,marker='o')
ax[1,3].plot(x,y,linewidth=1,color='darkgrey',linestyle='--')
ax[1,3].scatter(Logmstar_c_com,Conc_c_com,c=sfr_c_com, cmap=cm,vmin=-11,vmax=-9.5,s=12,marker='o')
ax[1,3].scatter(Logmstar_c_com,Conc_c_com,facecolors='none', edgecolors='r',s=35,marker='o',linewidth=0.5,alpha=0.5)
ax[1,3].errorbar(Logmstar_c_com,Conc_c_com, (Err_neg_c,Err_pos_c), fmt='none',ecolor='black', elinewidth=1.5,alpha=0.5)
ax[1,3].text(7.7, -0.58, '(h)', fontsize=12)


r1=mass_f+mass_f_2; hde1=conc_f+conc_f_2;
r2=mass_l+mass_l_2; hde2=conc_l+conc_l_2;
r3=mass_h+mass_h_2; hde3=conc_h+conc_h_2;
r4=mass_c+mass_c_2; hde4=conc_c+conc_c_2;
r5=mass_f+mass_f_2+mass_f_com; hde5=conc_f+conc_f_2+conc_f_com;
r6=mass_l+mass_l_2+mass_l_com; hde6=conc_l+conc_l_2+conc_l_com;
r7=mass_h+mass_h_2+mass_h_com; hde7=conc_h+conc_h_2+conc_h_com;
r8=mass_c+mass_c_2+mass_c_com; hde8=conc_c+conc_c_2+conc_c_com;

hhh =hde5+hde6+hde7+hde8
bins_num=4;xaxis_add=3/bins_num/2
range_min=8;range_max=11
bin_means_1, bin_edges_1, binnumber_1=stats.binned_statistic(r1, hde1, 'median', bins=bins_num,range=(range_min,range_max))
bin_means_2, bin_edges_2, binnumber_2=stats.binned_statistic(r2, hde2, 'median', bins=bins_num,range=(range_min,range_max))
bin_means_3, bin_edges_3, binnumber_3=stats.binned_statistic(r3, hde3, 'median', bins=bins_num,range=(range_min,range_max))
bin_means_4, bin_edges_4, binnumber_4=stats.binned_statistic(r4, hde4, 'median', bins=bins_num,range=(range_min,range_max))
bin_means_5, bin_edges_5, binnumber_5=stats.binned_statistic(r5, hde5, 'median', bins=bins_num,range=(range_min,range_max))
bin_means_6, bin_edges_6, binnumber_6=stats.binned_statistic(r6, hde6, 'median', bins=bins_num,range=(range_min,range_max))
bin_means_7, bin_edges_7, binnumber_7=stats.binned_statistic(r7, hde7, 'median', bins=bins_num,range=(range_min,range_max))
bin_means_8, bin_edges_8, binnumber_8=stats.binned_statistic(r8, hde8, 'median', bins=bins_num,range=(range_min,range_max))

bin_std_1, bin_edges_1, binnumber_1=stats.binned_statistic(r1, hde1, 'std', bins=bins_num,range=(range_min,range_max))
bin_std_2, bin_edges_2, binnumber_2=stats.binned_statistic(r2, hde2, 'std', bins=bins_num,range=(range_min,range_max))
bin_std_3, bin_edges_3, binnumber_3=stats.binned_statistic(r3, hde3, 'std', bins=bins_num,range=(range_min,range_max))
bin_std_4, bin_edges_4, binnumber_4=stats.binned_statistic(r4, hde4, 'std', bins=bins_num,range=(range_min,range_max))
bin_std_5, bin_edges_5, binnumber_5=stats.binned_statistic(r5, hde5, 'std', bins=bins_num,range=(range_min,range_max))
bin_std_6, bin_edges_6, binnumber_6=stats.binned_statistic(r6, hde6, 'std', bins=bins_num,range=(range_min,range_max))
bin_std_7, bin_edges_7, binnumber_7=stats.binned_statistic(r7, hde7, 'std', bins=bins_num,range=(range_min,range_max))
bin_std_8, bin_edges_8, binnumber_8=stats.binned_statistic(r8, hde8, 'std', bins=bins_num,range=(range_min,range_max))

bin_count_1, bin_edges_1, binnumber=stats.binned_statistic(r1, hde1, 'count', bins=bins_num,range=(range_min,range_max))
bin_count_2, bin_edges_2, binnumber=stats.binned_statistic(r2, hde2, 'count', bins=bins_num,range=(range_min,range_max))
bin_count_3, bin_edges_3, binnumber=stats.binned_statistic(r3, hde3,'count', bins=bins_num,range=(range_min,range_max))
bin_count_4, bin_edges_4, binnumber=stats.binned_statistic(r4, hde4, 'count', bins=bins_num,range=(range_min,range_max))
bin_count_5, bin_edges_5, binnumber=stats.binned_statistic(r5, hde5, 'count', bins=bins_num,range=(range_min,range_max))
bin_count_6, bin_edges_6, binnumber=stats.binned_statistic(r6, hde6, 'count', bins=bins_num,range=(range_min,range_max))
bin_count_7, bin_edges_7, binnumber=stats.binned_statistic(r7, hde7, 'count', bins=bins_num,range=(range_min,range_max))
bin_count_8, bin_edges_8, binnumber=stats.binned_statistic(r8, hde8, 'count', bins=bins_num,range=(range_min,range_max))

bin_means_1[np.where(bin_count_1<5)]=np.nan
bin_means_2[np.where(bin_count_2<5)]=np.nan
bin_means_3[np.where(bin_count_3<5)]=np.nan
bin_means_4[np.where(bin_count_4<5)]=np.nan
bin_means_5[np.where(bin_count_5<5)]=np.nan
bin_means_6[np.where(bin_count_6<5)]=np.nan
bin_means_7[np.where(bin_count_7<5)]=np.nan
bin_means_8[np.where(bin_count_8<5)]=np.nan


ax[0,0].errorbar(bin_edges_1[:-1]+xaxis_add,bin_means_1,yerr=bin_std_1/np.sqrt(bin_count_1),c='black',linestyle='-',linewidth=1.5, label='Median')
ax[0,1].errorbar(bin_edges_2[:-1]+xaxis_add,bin_means_2,yerr=bin_std_2/np.sqrt(bin_count_2),c='black',linestyle='-',linewidth=1.5)
ax[0,2].errorbar(bin_edges_3[:-1]+xaxis_add,bin_means_3,yerr=bin_std_3/np.sqrt(bin_count_3),c='black',linestyle='-',linewidth=1.5)
ax[0,3].errorbar(bin_edges_4[:-1]+xaxis_add,bin_means_4,yerr=bin_std_4/np.sqrt(bin_count_4),c='black',linestyle='-',linewidth=1.5)
ax[1,0].errorbar(bin_edges_5[:-1]+xaxis_add,bin_means_5,yerr=bin_std_5/np.sqrt(bin_count_5),c='black',linestyle='-',linewidth=1.5)
ax[1,1].errorbar(bin_edges_6[:-1]+xaxis_add,bin_means_6,yerr=bin_std_6/np.sqrt(bin_count_6),c='black',linestyle='-',linewidth=1.5,label='SF-concentrated galaxy')
ax[1,2].errorbar(bin_edges_7[:-1]+xaxis_add,bin_means_7,yerr=bin_std_7/np.sqrt(bin_count_7),c='black',linestyle='-',linewidth=1.5)
ax[1,3].errorbar(bin_edges_8[:-1]+xaxis_add,bin_means_8,yerr=bin_std_8/np.sqrt(bin_count_8),c='black',linestyle='-',linewidth=1.5)


ax[0,0].legend(loc='lower right',fontsize=10)

#plt.savefig('/Users/ice/Dropbox (Sydney Uni)/PHDprocess/dncode/plotAdam/paperimage/v0.12/CIndexvsMass_withoutBCG_errbar.pdf')

plt.show()
#plt.close()


