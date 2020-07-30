import numpy as np
import numba
import scipy.special as sp_spec
import scipy.integrate as sp_int
from scipy.optimize import minimize, curve_fit
import sys
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import quad
def pseq_params(params):
    Qe, Te, Ee = params['Qe'], params['Te'], params['Ee']
    Qi, Ti, Ei = params['Qi'], params['Ti'], params['Ei']
    Gl, Cm , El = params['Gl'], params['Cm'] , params['El']
    for key, dval in zip(['Ntot', 'pconnec', 'gei'], [1, 2., 0.5]):
        if key in params.keys():
            exec(key+' = params[key]')
        else: # default value
            exec(key+' = dval')
    if 'P' in params.keys():
        P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10 = params['P']
    else: # no correction
        P0 = -45e-3
        for i in range(1,11):
            exec('P'+str(i)+'= 0')

    return Qe,Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10

# @numba.jit()


def get_fluct_regime_varsup(Fe, Fi, XX,Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10):
    

    
    
    
    # here TOTAL (sum over synapses) excitatory and inhibitory input
    fe = Fe*(1.-gei)*pconnec*Ntot # default is 1 !!
    
    
    
    
    fi = Fi*gei*pconnec*Ntot
    muGe, muGi = Qe*Te*fe, Qi*Ti*fi
    muG = Gl+muGe+muGi
    muV = (muGe*Ee+muGi*Ei+Gl*El-XX)/muG
    muGn, Tm = muG/Gl, Cm/muG
    
    Ue, Ui = Qe/muG*(Ee-muV), Qi/muG*(Ei-muV)

    sV = np.sqrt(\
                 fe*(Ue*Te)**2/2./(Te+Tm)+\
                 fi*(Ti*Ui)**2/2./(Ti+Tm))

    fe, fi = fe+1e-9, fi+1e-9 # just to insure a non zero division,
   
    Tv = ( fe*(Ue*Te)**2 + fi*(Ti*Ui)**2 ) /( fe*(Ue*Te)**2/(Te+Tm) + fi*(Ti*Ui)**2/(Ti+Tm) )
    TvN = Tv*Gl/Cm

    return muV, sV+1e-12, muGn, TvN




def mean_and_var_conductance(Fe, Fi, Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10):
    # here TOTAL (sum over synapses) excitatory and inhibitory input
    fe = Fe*(1.-gei)*pconnec*Ntot # default is 1 !!
    fi = Fi*gei*pconnec*Ntot
    return Qe*Te*fe, Qi*Ti*fi, Qe*np.sqrt(Te*fe/2.), Qi*np.sqrt(Ti*fi/2.)


### FUNCTION, INVERSE FUNCTION
# @numba.jit()
def erfc_func(muV, sV, TvN, Vthre, Gl, Cm):
    return .5/TvN*Gl/Cm*(sp_spec.erfc((Vthre-muV)/np.sqrt(2)/sV))

# @numba.jit()
def effective_Vthre(Y, muV, sV, TvN, Gl, Cm):
    Vthre_eff = muV+np.sqrt(2)*sV*sp_spec.erfcinv(\
                    Y*2.*TvN*Cm/Gl) # effective threshold
    return Vthre_eff

# @numba.jit()
def threshold_func(muV, sV, TvN, muGn, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10):
    """
    setting by default to True the square
    because when use by external modules, coeff[5:]=np.zeros(3)
    in the case of a linear threshold
    """
    
    muV0, DmuV0 = -60e-3,10e-3
    sV0, DsV0 =4e-3, 6e-3
    TvN0, DTvN0 = 0.5, 1.
    
    return P0+P1*(muV-muV0)/DmuV0+\
        P2*(sV-sV0)/DsV0+P3*(TvN-TvN0)/DTvN0+\
        0*P4*np.log(muGn)+P5*((muV-muV0)/DmuV0)**2+\
        P6*((sV-sV0)/DsV0)**2+P7*((TvN-TvN0)/DTvN0)**2+\
        P8*(muV-muV0)/DmuV0*(sV-sV0)/DsV0+\
        P9*(muV-muV0)/DmuV0*(TvN-TvN0)/DTvN0+\
        P10*(sV-sV0)/DsV0*(TvN-TvN0)/DTvN0
      
# final transfer function template :
# @numba.jit()

def TF_my_templateup(fe, fi,XX, Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10):
    # here TOTAL (sum over synapses) excitatory and inhibitory input
    
    

    if(hasattr(fe, "__len__")):
        
        fe[fe<1e-8]=1e-8


    else:
    
        if(fe<1e-8):
            fe=1e-8

    if(hasattr(fi, "__len__")):
    
        fi[fi<1e-8]=1e-8

    
    else:
        
        if(fi<1e-8):
            fi=1e-8

   
   
   
   
   
    muV, sV, muGn, TvN = get_fluct_regime_varsup(fe, fi,XX, Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)
    Vthre = threshold_func(muV, sV, TvN, muGn, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)
    

    if(hasattr(muV, "__len__")):
        #print("ttt",isinstance(muV, list), hasattr(muV, "__len__"))
        sV[sV<1e-4]=1e-4
    
    
    else:
        
        if(sV<1e-4):
            sV=1e-4



    
    Fout_th = erfc_func(muV, sV, TvN, Vthre, Gl, Cm)
    if(hasattr(Fout_th, "__len__")):
        #print("ttt",isinstance(muV, list), hasattr(muV, "__len__"))
        Fout_th[Fout_th<1e-8]=1e-8
    
    
    else:
        
        if(Fout_th<1e-8):
            Fout_th=1e-8
    '''
    if(El<-0.063):
    
        if(hasattr(Fout_th, "__len__")):
            #print("ttt",isinstance(muV, list), hasattr(muV, "__len__"))
            Fout_th[Fout_th>80.]=175
    
    
        else:
        
            if(Fout_th>80.):
                print("Done")
                Fout_th=175

    '''
    #print 'yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy',fe,fi,muV, sV, TvN,Fout_th
    return Fout_th



def gaussian(x, mu, sig):
    return (1/(sig*np.sqrt(2*3.1415)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
def TF_my_templateup_heterogeneity(fe, fi,XX, Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10):
    # here TOTAL (sum over synapses) excitatory and inhibitory input
    
    
    
    
    def Phet(k):
        locale=gaussian(k,1.,0.2)*TF_my_templateup(fe, fi,XX, Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El*k, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)
        return locale


    
    outhet, err = quad(Phet, 0.1, 5)

    
    
  
    return outhet
    






    
# @numba.jit()
def make_loop(t, nu, vm, nu_aff_exc, nu_aff_inh, BIN,\
              Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10):
    dt = t[1]-t[0]
    # constructing the Euler method for the activity rate
    for i_t in range(len(t)-1): # loop over time
        
        fe = (nu_aff_exc[i_t]+nu[i_t]+Fdrive) # afferent+recurrent excitation
        fi = nu[i_t]+nu_aff_inh[i_t] # recurrent inhibition
        W[i_t+1] = W[i_t] + dt/Tw*(b*nu[i_t]*Tw - W[i_t])

        nu[i_t+1] = nu[i_t] +\
               dt/BIN*(\
                TF_my_template(fe, fi, W[i_t], Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)\
                -nu[i_t])

        vm[i_t], _, _, _ = get_fluct_regime_vars(fe, fi, W[i_t], Qe, Te, Ee, Qi, Ti, Ei, Gl, Cm, El, Ntot, pconnec, gei, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)

    return nu, vm, W


################################################################
##### Now fitting to Transfer Function data
################################################################


def fitting_Vthre_then_Fout(Fout, Fe_eff, fiSim,w, params,\
                               maxiter=50000, xtol=1e-5, with_square_terms=False):


    Gl, Cm , El = params['Gl'], params['Cm'] , params['El']

    Fout, Fe_eff, fiSim,w = Fout.flatten(), Fe_eff.flatten(), fiSim.flatten(), w.flatten()
    #print 'Eccolo', Fout[8],w[8],len(Fout)
    
    muV, sV, muGn, TvN = get_fluct_regime_varsup(Fe_eff, fiSim,w, *pseq_params(params))
    i_non_zeros = np.where((Fout>0.)&(Fout<60.))

    Vthre_eff = effective_Vthre(Fout[i_non_zeros], muV[i_non_zeros],\
                sV[i_non_zeros], TvN[i_non_zeros], params['Gl'], params['Cm'])
    
    if with_square_terms:
        P = np.zeros(11)
    else:
        P = np.zeros(5)
    P[:5] = Vthre_eff.mean(), 1e-3, 1e-3, 1e-3, 1e-3

    def Res(p):
        if not with_square_terms:
            pp = np.concatenate([p, np.zeros(6)])
        else:
            pp=p
        vthre = threshold_func(muV[i_non_zeros], sV[i_non_zeros],\
                               TvN[i_non_zeros], muGn[i_non_zeros], *pp)
        return np.mean((Vthre_eff-vthre)**2)
            
    #xtol=1e-19
    #plsq = minimize(Res, P, method='nelder-mead',options={'xtol': xtol, 'disp': True, 'maxiter':maxiter})
    plsq = minimize(Res, P, method='SLSQP',options={'ftol': 1e-15, 'disp': True, 'maxiter':40000})
    #print plsq

    P = plsq.x
    
    def Res(p):
        if not with_square_terms:
            params['P'] = np.concatenate([p, np.zeros(6)])
        else:
            params['P'] = p
        return np.mean((Fout-\
                        TF_my_templateup(Fe_eff, fiSim,w, *pseq_params(params)))**2)

    plsq = minimize(Res, P, method='nelder-mead',\
            options={'xtol': xtol, 'disp': True, 'maxiter':maxiter})



    params['P'] = P
    
    diff=(TF_my_templateup(Fe_eff, fiSim,w, *pseq_params(params))-Fout).mean()
    diff_M=(TF_my_templateup(Fe_eff, fiSim,w, *pseq_params(params))-Fout).max()
    print("rrrrr",diff,diff_M)
          
          
  
    plt.plot(fiSim,Fout,'rd',fiSim,TF_my_templateup(Fe_eff, fiSim,w, *pseq_params(params)),'bs')
    #plt.plot(fiSim,Fe-eff,Fout,'rd')
    plt.show()
    thrplot=threshold_func(muV, sV,TvN, muGn, *(plsq.x))
    
    #np.save('FScell_Voltage.npy',[muV, sV,TvN,Fout])
   
    plt.plot(muV,Fout,'rd',muV,erfc_func(muV, sV, TvN, thrplot, Gl, Cm),'bd')
    plt.show()
    plt.plot(muV,erfc_func(muV, 4e-3, 0.5, thrplot, Gl, Cm),'bd')
    plt.show()



    if with_square_terms:
        #return plsq.x
        return P
    else:
        return np.concatenate([plsq.x, np.zeros(6)])

def make_fit_from_data(DATA, with_square_terms=False):

    MEANfreq, SDfreq, Fe_eff, fiSim, params,w = np.load(DATA)
    

    Fe_eff, Fout = np.array(Fe_eff), np.array(MEANfreq)
    levels = fiSim # to store for colors
    fiSim = np.meshgrid(np.zeros(Fe_eff.shape[1]), fiSim)[1]

    P = fitting_Vthre_then_Fout(Fout, Fe_eff, fiSim,w, params,\
                                with_square_terms=with_square_terms)
                                
    print("ffffff",P)

    #plt.plot(Fe_eff[2,:],MEANfreq[2,:],"bs",fiSim[:,5],MEANfreq[:,5],"o")
    plt.plot(Fe_eff[2,:],MEANfreq[2,:],"bs",fiSim[:,5],MEANfreq[:,5],"o")
    #np.save('FScell_freq.npy',[Fe_eff[2,:],fiSim[:,5],MEANfreq[:,5]])

    plt.show()
                            
    print '=================================================='
    print 1e3*np.array(P), 'mV'

    # then we save it:
    filename = DATA.replace('.npy', '_fit.npy')
    print 'coefficients saved in ', filename
    np.save(filename, np.array(P))

    return P

    
import argparse
if __name__=='__main__':
    # First a nice documentation 
    parser=argparse.ArgumentParser(description=
     """ 
     '=================================================='
     '=====> FIT of the transfer function =============='
     '=== and theoretical objects for the TF relation =='
     '=================================================='
     """,
              formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-f', "--FILE",help="file name of numerical TF data",\
                        default='data/example_data.npy')
    parser.add_argument("--With_Square",help="Add the square terms in the TF formula"+\
                        "\n then we have 7 parameters",\
                         action="store_true")
    args = parser.parse_args()

    make_fit_from_data(args.FILE, with_square_terms=args.With_Square)
