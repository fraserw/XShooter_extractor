from bgRem import *
import glob,sys, pickle as pick
from astropy.io import fits
import pylab as pyl, numpy as np
from stsci import numdisplay
from astropy.visualization import interval
from scipy import interpolate as interp

showPlots = False

if len(sys.argv)>1:
    fn = sys.argv[1]
else:
    fn = 'AllOrders/DR30_SCI_SLIT_ORDER2D_NIR_6.fits'
    #fn = 'AllOrders/HD97356_SCI_SLIT_ORDER2D_NIR_3.fits'

print 'Processing',fn
print
han = fits.open(fn)

#get the skip region using order 5, index 12
data = han[12].data
errs = han[13].data
qual = han[14].data
header = han[12].header
waves = np.arange(data.shape[1])*header['CDELT1']  + header['CRVAL1']
(A,B) = data.shape

centFit = centroidFit(data-np.median(data[4:-3,:]), qual)
centFit(B, verbose = True, useTwoSidedMoffat =  True)

coeffs = centFit.coeffs
fullCentGuess = coeffs[0][1]
x = centFit.x
moffat_profiles,FWHM = centFit.moffat_profiles

#ul = (coeffs[:,1]+FWHM*2)[0]
#ll = (coeffs[:,1]-FWHM*2)[0]
#print ll,ul
#print moffat_profiles[0][ul]/np.max(moffat_profiles[0])
#print moffat_profiles[0][ll]/np.max(moffat_profiles[0])
#pyl.plot(moffat_profiles[0])
#pyl.scatter(np.arange(len(moffat_profiles[0])),np.median(data-np.median(data),axis=1))
#pyl.show()
#sys.exit()

if 'HD' in fn:#np.nanmax(data)>1.e5:
    ul = (coeffs[:,1]+FWHM*3)[0]
    ll = (coeffs[:,1]-FWHM*3)[0]
else:
    ul = (coeffs[:,1]+FWHM*2)[0]
    ll = (coeffs[:,1]-FWHM*2)[0]

default_skip =[int(ll),int(ul)]

print 'Skipping ',default_skip,FWHM[0]
print

fwhmWidth = np.linspace(1.0,2.0,5)

polyOrders = [4, 4, 4, 4, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4]

pars = [default_skip[0], default_skip[1], FWHM[0], fwhmWidth]
out_coeffs = []
out_FWHM = []
HDUs = []
specs = []
waves = []

if not showPlots:
    FWHMfig = pyl.figure('Moffats',figsize=(15,15))

for i in range(0,len(han),3):
    order = i/3

    if order == 0:
        skip = [default_skip[0] - int(FWHM), default_skip[1] + int(FWHM)]
    else:
        skip = default_skip[:]

    if not showPlots:
        fsp = FWHMfig.add_subplot(4,4,order+1)

    moffSampNum = 6
    #if order in [0,1]:
    #    moffSampNum = 5
    if order in [15]:
        moffSampNum = 4

    polyOrder = polyOrders [order]
    edgeSkip = 3

    pars.append(moffSampNum)
    pars.append(polyOrder)

    print
    print polyOrder,moffSampNum
    print 'Order {} with polyOrder {} and moffSampNum {}'.format(i/3,polyOrder,moffSampNum)

    data = np.copy(han[i].data)
    errs = np.copy(han[i+1].data)
    qual = np.copy(han[i+2].data)
    header = han[i].header
    wave = np.arange(data.shape[1])*header['CDELT1']  + header['CRVAL1']

    if i==0:
        HDUs.append(fits.PrimaryHDU(data,header=header))
    else:
        HDUs.append(fits.ImageHDU(data,header=header))
    HDUs.append(fits.ImageHDU(errs))
    HDUs.append(fits.ImageHDU(qual))

    (A,B) = data.shape



    header = han[i].header

    #print waves,'%%'
    #med = 1.e-18#np.median(data)
    #data /= med
    #data*=1000.0




    bg = []
    for i in range(B):
        f = data[:,i]
        q = qual[:,i]
        p = bgPoly(np.arange(len(f))+1.0, f, skip, bpMask = q)
        y = p(polyOrder)
        bg.append(y)
    bg = np.array(bg).T
    HDUs.append(fits.ImageHDU(bg))

    rem = data - bg
    HDUs.append(fits.ImageHDU(rem))


    (z1,z2)=numdisplay.zscale.zscale(data)
    normer=interval.ManualInterval(z1,z2)
    #pyl.imshow(normer(data), interpolation='nearest')#,origin='lower')

    """
    print data
    print rem
    print bg
    fig1 = pyl.figure(1)
    pyl.imshow(normer(data), interpolation='nearest')#,origin='lower')
    fig1 = pyl.figure(2)
    pyl.imshow(normer(bg), interpolation='nearest')#,origin='lower')
    fig1 = pyl.figure(3)
    pyl.imshow(normer(rem), interpolation='nearest')#,origin='lower')
    pyl.show()
    sys.exit()
    """

    #qual[ 0,:] = 1.0
    #qual[-1,:] = 1.0

    #no fit moffat profiles and get the apertures
    passed = []
    while len(passed)< moffSampNum:
        centFit = centroidFit(rem, qual)
        centFit(B/moffSampNum, verbose = True, useTwoSidedMoffat =  True)#, centGuess = fullCentGuess)
        passed = centFit.passed[:]
        if len(passed) < moffSampNum:
            print 'Rerunning with lower Moffat Sample Number'
            moffSampNum -= 1
        if moffSampNum == 1:
            print 'Still Failed'
            sys.exit()

    coeffs = centFit.coeffs
    x = centFit.x
    moffat_profiles,FWHM = centFit.moffat_profiles
    fs = centFit._fs
    print 'FWHM:', FWHM
    print np.median(FWHM), np.std(FWHM)
    bad_FWHM = np.where(np.abs(FWHM-np.median(FWHM))>2*np.std(FWHM))[0]
    FWHM[bad_FWHM] = np.median(FWHM)
    f_profiles = centFit.f_profiles

    out_FWHM.append(FWHM)
    out_coeffs.append(coeffs)



    #now use the moffat profiles to interpolate over bad pixels
    #also get the pixel-sampled apertures
    xx = np.arange(B)
    pp_moffat_profiles = []
    for i in range(A):
        fm = betterInterp(x, moffat_profiles[:,i])
        pp_moffat_profiles.append(fm(np.arange(B)))
    pp_moffat_profiles = np.array(pp_moffat_profiles)

    e_qual = np.copy(qual)
    expandQualFlags = False
    #try expanding the quality flags to +-1 of bad
    if expandQualFlags:
        w = np.where(qual>0)
        for ii in range(len(w[0])):
            x,y = w[0][ii],w[1][ii]
            e_qual[max(x-1,0),y] = qual[x,y]
            e_qual[min(x+1,A-1),y] = qual[x,y]
            e_qual[x,max(y-1,0)] = qual[x,y]
            e_qual[x,min(y+1,B-1)] = qual[x,y]

    e_qual[0,:] = 1.0
    e_qual[1,:] = 1.0
    e_qual[-1,:] = 1.0
    e_qual[-2,:] = 1.0
    w = np.where(e_qual>0)
    rem[w] = pp_moffat_profiles[w]


    #now extract a spectrum
    s = np.zeros((len(fwhmWidth),B)).astype('float64')
    repFact = 20.0

    for jj in range(len(fwhmWidth)):
        upperLims = coeffs[:,1]+FWHM*fwhmWidth[jj]/2.0+1
        lowerLims = coeffs[:,1]-FWHM*fwhmWidth[jj]/2.0+1

        f = betterInterp(x,coeffs[:,1]+1)
        fu = betterInterp(x,upperLims)
        fl = betterInterp(x,lowerLims)


        pp_upperLims = fu(xx)
        pp_lowerLims = fl(xx)


        #if jj == len(fwhmWidth)-1:
        #    for qrt in range(len(upperLims)):
        #        print lowerLims[qrt],upperLims[qrt]
        #        pyl.plot(fs[qrt][int(lowerLims[qrt]):int(upperLims[qrt])])
        #        pyl.show()
        #    sys.exit()


        xr = np.arange(A*int(repFact))/float(repFact)
        for ii in range(B):
            fr = np.repeat(rem[:,ii],int(repFact))/float(repFact)
            w = np.where((xr>pp_lowerLims[ii]) & (xr<pp_upperLims[ii]))
            s[jj,ii] = np.sum(fr[w])

    specHDU = fits.ImageHDU(s)
    HDUs.append(specHDU)



    if showPlots:

        fig1 = pyl.figure('Order',figsize=(32,8))
        pyl.imshow(normer(rem))
        pyl.plot(xx,fu(xx))
        pyl.plot(xx,fl(xx))
        pyl.plot(xx,f(xx),'k--',lw=2)

        fig2 = pyl.figure('Spectrum',figsize=(8,8))
        pyl.title('Order {}'.format(order))
        for jj in range(len(s)):
            pyl.plot(wave,s[jj])

        fig3 = pyl.figure('Moffat Profiles',figsize=(8,8))
        for ii in range(len(moffat_profiles)):
            l = pyl.plot(moffat_profiles[ii])
            pyl.scatter(np.arange(len(f_profiles[ii]))[2:-2],f_profiles[ii][2:-2],c=l[0].get_color())
            pyl.title('Order {}'.format(order))

        pyl.show()
        #sys.exit()
        pyl.close(fig1)
        pyl.close(fig2)
        pyl.close(fig3)

    elif not showPlots:
        for ii in range(len(moffat_profiles)):
            l = fsp.plot(moffat_profiles[ii])
            fsp.scatter(np.arange(len(f_profiles[ii]))[2:-2],f_profiles[ii][2:-2],c=l[0].get_color())
            fsp.set_title('Order={} MSN={} PO={} '.format(order, moffSampNum,polyOrder))

    #sys.exit()

    specs.append(np.copy(s))
    waves.append(np.copy(wave))

with open(fn.replace('.fits','_bgr.spectra'),'w+') as outhan:
    pick.dump([waves, specs, pars, out_coeffs, out_FWHM],outhan)

out = fits.HDUList(HDUs)
out.writeto(fn.replace('.','_bgr.'), clobber = True)

if not showPlots:
    FWHMfig.savefig(fn.replace('.fits','_bgr.pdf'),bbox='tight')
