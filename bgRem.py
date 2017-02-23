from astropy.io import fits
from trippy import bgFinder
import numpy as np,sys
from numpy import linalg
import pylab as pyl
from scipy.optimize import curve_fit
from scipy import interpolate as interp


def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def moffat(x, *p):
    # Creates moffat profile.
    A, c, alpha, gamma = p
    moff = moffat_pars(x, A, c, alpha, gamma)
    return moff

def moffatTwoSided(x, *p):
    # Creates moffat profile.
    A, c, alphal, gammal, alphar, gammar = p
    moff = moffat_pars(x, A, c, alphal, gammal)
    w = np.where(x>c)
    moff[w] = moffat_pars(x[w], A, c, alphar, gammar)
    return moff

def moffat_pars(x, A,c,alpha,gamma):
    xmc = x-c
    moff = A * np.power(1 + np.divide((xmc * xmc), gamma * gamma), -alpha)
    return moff

#x = np.arange(52)
#pyl.plot(x, moffat_pars(x, 100.0,25.2,2.0,2.0 ) )
#pyl.show()
#sys.exit()

class bgPoly:
    """
    Will skip the highest pixel in the spectrum because that's obviously bad.
    """

    def __init__(self, x, y, skip, maxOrders=5, bpMask = None, edgeSkip = 3):
        """
        bad pixel mask takes non-zero values to mark BAD pixels
        """

        self._xo = np.copy(x)
        self._yo = np.copy(y)

        self._skip = skip
        self.edgeSkip = edgeSkip


        if bpMask is not None:
            self._bp = np.copy(bpMask)
        else:
            self._bp = np.zeros(len(x))

        if self._skip is not None:
            for ii in range(skip[0],skip[1]):
                self._bp[ii] = 1.0
        for ii in range(edgeSkip+1):
            self._bp[ii] = 1.0
        for ii in range(edgeSkip):
            self._bp[-ii] = 1.0

        w = np.where(self._bp==0)

        xs = x[w] - 0.0
        ys = y[w]

        #old version which doesn't use the bad pixel mask
        self._x = []
        self._y = np.array(ys)
        self.maxOrders = maxOrders

        for n in range(self.maxOrders+1):
            self._x.append(xs**n)
        self._x = np.array(self._x).T

        self._a = []
        for o in range(self.maxOrders+1):
            if len(xs) == 0:
                q = np.zeros(o+1).astype('float64')
                q[0] += 0.0
                self._a.append(q)
                continue
            elif len(xs)<2*(o+1):
                q = np.zeros(o+1).astype('float64')
                q[0] += np.median(ys)
                self._a.append(q)
                continue

            #o = 3
            A = self._x[:,:o+1]
            At = A.T
            AtA = np.dot(At,A)
            AtAinv = linalg.inv(AtA)
            AtAiAt = np.dot(AtAinv,At)

            self._a.append(np.dot(AtAiAt,self._y))
            #print self._a[-1]

    def __call__(self, order):
        self.Y = self._a[order][0]*self._xo*0.0
        for n in range(order+1):
            self.Y += self._a[order][n]*(self._xo - 0.0)**n
        return self.Y

    def show(self,x = None, y = None):
        pyl.plot(self._xo,self._yo)
        pyl.plot(self._xo,self.Y,'ks-')

        if x is not None and y is not None:
            pyl.plot(x,y[self.edgeSkip+1:-self.edgeSkip],'r-',lw = 2)
        mini,maxi = np.min(self._yo[self.edgeSkip+1:-self.edgeSkip]), np.max(self._yo[self.edgeSkip+1:-self.edgeSkip])
        pyl.axis([-1,53,mini,maxi])
        pyl.show()


class centroidFit:
    def __init__(self, spectrum, bpMask):
        self._spec = np.copy(spectrum)
        self._bp = np.copy(bpMask)
        self._coeffs = []
        self._x = []
        self._fs = []
        self._ft = []
        self._passed = []

    def __call__(self, binWidth = 10,
                 verbose = False, showPlot = False,
                 useMedian = False, useMoffat = False, useTwoSidedMoffat = False,
                 edgeSkip = 3, centGuess = None):
        self._edgeSkip = edgeSkip
        self._fitForm = 'gaussian'
        if useMedian:
            self._fitForm = 'median'
        elif useMoffat:
            self._fitForm = 'moffat'
        elif useTwoSidedMoffat:
            self._fitForm = 'moffat2'

        passed = []
        (A,B) = self._spec.shape
        for ii in range(0, B, int(binWidth)):
            if ii>B or ii+int(binWidth)>B: break
            sec = self._spec[:,ii:ii+int(binWidth)]
            bp = self._bp[:,ii:ii+int(binWidth)]
            w = np.where(bp>0)

            sec[w] = np.nan
            f = np.nanmedian(sec,axis=1)
            f_trim = np.copy(f[edgeSkip+1:-edgeSkip])
            f_trim -= np.nanmedian(f)

            if showPlot:
                pyl.clf()
                pyl.plot(f_trim)
                pyl.title('Order {}'.format(ii/int(binWidth)))
                pyl.show()

            self._fs.append(np.copy(f))
            self._ft.append(np.copy(f_trim))

            #pyl.plot(f_trim)
            #pyl.show()

            if useMedian:
                self._coeffs.append([np.max(f), np.argmax(f), -1.0])
                passed.append(len(self._coeffs)-1)
                continue

            if useMoffat:
                try:
                    p = [np.max(f_trim), np.argmax(f_trim), 2.0, 3.96]
                    coeff, var_matrix = curve_fit(moffat, np.arange(len(f_trim)), f_trim, p0=p)
                    self._coeffs.append(coeff)
                    if verbose: print coeff
                    passed.append(len(self._coeffs)-1)
                    """
                    m = moffat_pars(np.arange(len(f_trim)),coeff[0],coeff[1],coeff[2],coeff[3])
                    pyl.plot(np.arange(len(f_trim)), f_trim)
                    pyl.scatter(np.arange(len(f_trim)),m)
                    print coeff
                    pyl.show()
                    """
                    #sys.exit()
                except:
                    self._coeffs.append([-1.0,-1.0,-1.0,-1.0])
                    if verbose: print ii,'failed'
                    sys.exit()
                    #print f_trim.shape,f_trim
                    #pyl.plot(np.arange(len(f_trim)), f_trim)
                    #pyl.title('{} {}'.format(ii,ii+int(binWidth)))
                    #print np.max(f_trim), np.argmax(f_trim)
                    #pyl.show()
                    #sys.exit()
                continue

            if useTwoSidedMoffat:
                try:
                    p = [np.max(f_trim), np.argmax(f_trim), 2.0, 3.96, 2.0, 3.96]
                    if centGuess is not None:
                        p[1] = centGuess - edgeSkip -1
                    coeff, var_matrix = curve_fit(moffatTwoSided, np.arange(len(f_trim)), f_trim, p0=p)
                    self._coeffs.append(coeff)
                    if verbose: print coeff
                    passed.append(len(self._coeffs)-1)
                    """
                    m = moffat_pars(np.arange(len(f_trim)),coeff[0],coeff[1],coeff[2],coeff[3])
                    pyl.plot(np.arange(len(f_trim)), f_trim)
                    pyl.scatter(np.arange(len(f_trim)),m)
                    print coeff
                    pyl.show()
                    """
                    #sys.exit()
                except:
                    self._coeffs.append([-1.0,-1.0,-1.0,-1.0, -1.0, -1.0])
                    if verbose: print ii,'failed'
                    #print f_trim.shape,f_trim
                    #pyl.plot(np.arange(len(f_trim)), f_trim)
                    #pyl.title('{} {}'.format(ii,ii+int(binWidth)))
                    #print np.max(f_trim), np.argmax(f_trim)
                    #pyl.show()
                    #sys.exit()
                continue

            try:
                #fit a guassian to determine the approximate slit location of the source
                p = [np.max(f), np.argmax(f), 3.0]
                coeff, var_matrix = curve_fit(gauss, np.arange(len(f_trim)), f_trim, p0=p)
                if verbose: print ii + binWidth/2.0, coeff
                self._coeffs.append(coeff)
                passed.append(len(self._coeffs)-1)
            except:
                self._coeffs.append([-1.0,-1.0,-1.0])
                if verbose: print ii,'failed'


        self._coeffs = np.array(self._coeffs)
        self._x = np.arange(len(self._coeffs))*binWidth + int(binWidth)/2.0
        self._passed = np.array(passed)
        #print len(self._x),len(self._coeffs),len(passed)
        #self._x = self._x[passed]
        #self._coeffs = self._coeffs[passed]

        #fail replace hack
        for ii in range(len(self._coeffs)-1):
            if ii not in passed:
                print 'hack'
                self._coeffs[ii,:] = self._coeffs[ii+1,:]
        if len(self._coeffs)-1 not in passed:
            self._coeffs[-1,:] = self._coeffs[-2,:]


    @property
    def passed(self):
        return self._passed
    @property
    def coeffs(self):
        d = self._coeffs
        d[:,1] += self._edgeSkip+1
        return d
    @property
    def x(self):
        return self._x

    @property
    def moffat_profiles(self):
        print self._fitForm
        if self._fitForm not in ['moffat', 'moffat2']:
            return None

        (A,B) = self._spec.shape
        moff_prof = []
        FWHM = []
        if self._fitForm == 'moffat2':
            for ii in range(len(self._coeffs)):
                coeff = self._coeffs[ii]

                mof = moffat_pars(np.arange(A),coeff[0],coeff[1],coeff[2],coeff[3])
                w = np.where(np.arange(A)>coeff[1])
                mof[w] = moffat_pars(np.arange(A)[w],coeff[0],coeff[1],coeff[4],coeff[5])
                moff_prof.append(mof)

                xxx = np.arange(A*20)/20.0
                mof = moffat_pars(xxx,coeff[0],coeff[1],coeff[2],coeff[3])
                w = np.where(xxx>coeff[1])
                mof[w] = moffat_pars(xxx[w],coeff[0],coeff[1],coeff[4],coeff[5])
                high_res = mof
                #print coeff
                #pyl.plot(np.arange(A*20)/20.0,high_res)
                #pyl.show()
                w = np.where(high_res>0.5*np.max(high_res))[0]
                FWHM.append((np.max(w) - np.min(w))/20.0)
            return np.array(moff_prof), np.array(FWHM)
        else:
            for ii in range(len(self._coeffs)):
                coeff = self._coeffs[ii]
                moff_prof.append(moffat_pars(np.arange(A),coeff[0],coeff[1],coeff[2],coeff[3])) #+1 to handle edge trimming

                high_res = moffat_pars(np.arange(A*20)/20.0,coeff[0],coeff[1],coeff[2],coeff[3])
                #print coeff
                #pyl.plot(np.arange(A*20)/20.0,high_res)
                #pyl.show()
                w = np.where(high_res>0.5*np.max(high_res))[0]
                FWHM.append((np.max(w) - np.min(w))/20.0)
            return np.array(moff_prof), np.array(FWHM)

    @property
    def f_profiles(self):
        return self._fs

class betterInterp:
    def __init__(self,x,y,kind = 'cubic'):
        if len(x)<4:
            print 'recommend using quadratic or slinear'
        self._f = interp.interp1d(x,y,kind = kind)
        self._x = np.copy(x)
        self._y = np.copy(y)

    def __call__(self,x):
        wl = np.where(x < np.min(self._x))
        wh = np.where(x > np.max(self._x))
        w = np.where((x >= np.min(self._x))& (x <= np.max(self._x)))

        x_resamp = np.linspace(x[w][0],x[w][-1],len(w[0])*3)
        ml = (self._f(x_resamp[1]) - self._f(x_resamp[0])) / (x_resamp[1] - x_resamp[0])
        bl = self._f(x_resamp[1]) - ml*x_resamp[1]
        mh = (self._f(x_resamp[-2]) - self._f(x_resamp[-1])) / (x_resamp[-2] - x_resamp[-1])
        bh = self._f(x_resamp[-1]) - mh*x_resamp[-1]

        y = np.concatenate([ ml*x[wl] + bl, self._f(x[w]), mh*x[wh] + bh])
        return y






if __name__ == "__main__":
    default_skip = [17,33] #good for middle NIR, best order = 2
    #skip = [13,48] #good for middle UVB, best order = 1
    #skip = [13,56] #good for middle UVB, best order = 1

    fitsFile = 'AllOrders/DR30_SCI_SLIT_ORDER2D_NIR_1.fits'
    o = 15
    with fits.open(fitsFile) as han:
        data = han[o*3].data
        errs = han[o*3+1].data
        qual = han[o*3+2].data
        header = han[0].header


    if o >0: skip = default_skip[:]
    else:
        skip = default_skip[:]
        skip[0]-=4
        skip[1]+=4
    #skip = default_skip
    print skip
    (A,B) = data.shape

    #order 0, polyOrder = 4, edgeskip=3 seems best
    #order 1, polyORder = 4, edgeskip=4
    #order 2, polyORder = 4, edgeskip=4
    #order 3, polyORder = 4, edgeskip=4
    #order 4, polyORder = 3, edgeskip=4 #polrOrder>3 seems to overfit
    #order 5, polyORder = 2, edgeskip=4 #polrOrder>3 seems to overfit
    #order 6, polyORder = 2, edgeskip=4 #polrOrder>3 seems to overfit
    #order 7, polyORder = 2, edgeskip=4 #polrOrder>3 seems to overfit
    #order 8, polyORder = 3, edgeskip=4
    #order 9, polyORder = 3, edgeskip=4
    #order 10, polyORder = 3, edgeskip=4
    #order 11, polyORder = 3, edgeskip=4
    #order 12, polyORder = 3, edgeskip=4
    #order 13, polyORder = 3, edgeskip=4
    #order 14, polyORder = 4, edgeskip=4
    #order 15, polyORder = 4, edgeskip=4
    showMed = True
    if showMed:
        for ii in range(0,B,B/6):
            edgeSkip = 3
            polyOrder = 4
            a = ii
            b = a+B/6
            f = np.median(data[:,a:b],axis=1)
            bp = np.median(qual[:,a:b],axis=1)
            p = bgPoly(np.arange(len(f))+1.0, f, skip, bpMask=bp, edgeSkip = edgeSkip)
            y = p(polyOrder)

            #p.show()
            #continue

            ym = p(polyOrder-1)
            yM = p(polyOrder-2)
            pyl.plot((f-y)[edgeSkip:-edgeSkip])
            pyl.plot((f-ym)[edgeSkip:-edgeSkip])
            pyl.plot((f-yM)[edgeSkip:-edgeSkip])
            pyl.title(str(ii)+' '+str(polyOrder))
            pyl.show()
        sys.exit()

    bg = []
    for i in range(B):
        f = data[:,i]
        p = bgPoly(np.arange(len(f))+1.0, f, skip)
        y = p(2)
        bg.append(y)


    doCentroidFitting = True
    if doCentroidFitting:
        centFit = centroidFit(data[:,21345:21360]-np.array(bg).T[:,21345:21360], qual[:,21345:21360])
        centFit(3, verbose = True)
        pyl.plot(centFit.x,centFit.coeffs[:,1])
        pyl.plot(centFit.x,centFit.coeffs[:,2])
        pyl.show()

        sys.exit()

    bg = np.array(bg).T
    bg /= 1000.0
    bg *= med

    data /= 1000.0
    data *= med

    rem = data - bg

    HDU = fits.PrimaryHDU(data,header)
    IHDU = fits.ImageHDU(bg)
    RHDU = fits.ImageHDU(rem)

    List = fits.HDUList([HDU,IHDU,RHDU])
    List.writeto(fitsFile.replace('.','_bgr.'), overwrite = True)
    sys.exit()

    div = 50
    for i in range(0,B,B/div):

        f = np.median(data[:,i:i+B/div],axis=1)

        print poly(np.arange(len(f))+1.0, f, skip)
        continue
        sys.exit()
        #g = np.std(data[:,i:i+B/20],axis=1)*(B/20)**-0.5
        #print g/f
        #sys.exit()
        bgf = bgFinder.bgFinder(f)
        fraser =  bgf.fraserMode()
        median = np.median(f)
        pyl.plot([0,len(f)],[fraser,fraser],'r-')
        pyl.plot([0,len(f)],[median,median],'k-')
        pyl.plot(f)
        #pyl.plot(f+g,'k--')
        #pyl.plot(f-g,'k--')
        pyl.title(i+B/40)
        pyl.show()
