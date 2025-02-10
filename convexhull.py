#%%
## this fixes troubles with graphlab path...
################################################################################
# import sys
# print(sys.path)

# gl-env (to work with graphlab in python2)
#sys.path = ['', '/Users/chinchay/gl-env/bin', '/Users/chinchay/gl-env/lib/python27.zip', '/Users/chinchay/gl-env/lib/python2.7', '/Users/chinchay/gl-env/lib/python2.7/plat-darwin', '/Users/chinchay/gl-env/lib/python2.7/plat-mac', '/Users/chinchay/gl-env/lib/python2.7/plat-mac/lib-scriptpackages', '/Users/chinchay/gl-env/lib/python2.7/lib-tk', '/Users/chinchay/gl-env/lib/python2.7/lib-old', '/Users/chinchay/gl-env/lib/python2.7/lib-dynload', '/usr/local/Cellar/python@2/2.7.15_2/Frameworks/Python.framework/Versions/2.7/lib/python2.7', '/usr/local/Cellar/python@2/2.7.15_2/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-darwin', '/usr/local/Cellar/python@2/2.7.15_2/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-tk', '/usr/local/Cellar/python@2/2.7.15_2/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-mac', '/usr/local/Cellar/python@2/2.7.15_2/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-mac/lib-scriptpackages', '/Users/chinchay/gl-env/lib/python2.7/site-packages', '/Users/chinchay/gl-env/lib/python2.7/site-packages/IPython/extensions', '/Users/chinchay/.ipython']

# this brew installation of scipy (brew install scipy) works with python2. However `pip2 install scipy` gives
# segmentation error when using `ConvexHull`...
#sys.path.append('/usr/local/Cellar/scipy/1.2.0_1/lib/python2.7/site-packages/')


# print(sys.path)

################################################################################

import os
print os.getcwd()
# os.chdir('/home/leon/old3/2_codes/CoLiSn_withPrototypes/withPrototypesDftRelaxed/')
print os.getcwd()
#%%

# import graphlab
################################################################################


import sys
# sys.path.append('xPythonPackages/')

# import graphlab
import numpy as np

import CalcScripts as scr
import CalcScripts2 as sc2

#%%
################################################################################

#fileName = "/Users/chinchay/Documents/2_codes/CoLiSn/convexHull/relaxed.cfg"
fileName = "train2_joined.cfg"
# outFile  = "dat.csv"

conf_idL = sc2.getConfid(fileName)
nAlist, nBlist, nClist = sc2.getComposition(fileName)
energies = sc2.getEnergy(fileName)
mindistL = sc2.getMindist(fileName)
a1xL, a1yL, a1zL, a2xL, a2yL, a2zL, a3xL, a3yL, a3zL = sc2.getLatticeVector(fileName)


A = 'Co'
B = 'Li'
C = 'Sn'

assert len(nAlist) == len(conf_idL)
assert len(nBlist) == len(conf_idL)
assert len(nClist) == len(conf_idL)
assert len(energies) == len(conf_idL)
assert len(mindistL) == len(conf_idL)
assert len(a1xL) == len(conf_idL)
assert len(a1yL) == len(conf_idL)
assert len(a1zL) == len(conf_idL)
assert len(a2xL) == len(conf_idL)
assert len(a2yL) == len(conf_idL)
assert len(a2zL) == len(conf_idL)
assert len(a3xL) == len(conf_idL)
assert len(a3yL) == len(conf_idL)
assert len(a3zL) == len(conf_idL)
#%%
### ...

# head = "conf_id, " + A + ", " + B + ", " + C +\
#        ", totalEnergy, mindist, a1x, a1y, a1z, a2x, a2y, a2z, a3x, a3y, a3zL"
# with open(outFile, 'w') as myfile:
#     myfile.write( head )
#     for i in range(len(energies)):
#         line = str(conf_idL[i]) + str(nAlist) + str(nBlist) + str(nClist)
#         myfile.write( head )

################################################################################


t = sc2.buildTable(conf_idL, nAlist, nBlist, nClist, energies, mindistL,\
                a1xL, a1yL, a1zL,\
                a2xL, a2yL, a2zL,\
                a3xL, a3yL, a3zL,\
                A, B, C )

t

#%%
t.columns

#%%
##Plot of convex hull

import math
t['X'] = 100.0 * t[A] / (t[A] + t[B] + t[C])
t['Y'] = 100.0 * t[B] / (t[A] + t[B] + t[C])

t['Xt'] = t['X'] + ( 0.5 * t['Y'] )
t['Yt'] = t['Y'] * (math.sqrt(3.0) / 2.0)

t



def getChemLabel(colA, colB, colC):
    colChemFormula = []
    for i in range(len(colA)):
        a = colA[i]
        b = colB[i]
        c = colC[i]
        chemFormula = ""

        if (a != 0):
            if (a == 1):
                chemFormula += A
            else:
                chemFormula += A + str(a)
        if (b != 0):
            if (b == 1):
                chemFormula += B
            else:
                chemFormula += B + str(b)
        if (c != 0):
            if (c == 1):
                chemFormula += C
            else:
                chemFormula += C + str(c)
        #
        colChemFormula.append(chemFormula)

    #
    return colChemFormula

def getChemColum(A, B, C):
    return lambda t: getChemLabel( t, A, B, C )
#

# t['chemLabel'] = t.apply( getChemColum(A, B, C) )
# t['chemLabel'] = t.map( getChemColum(A, B, C) )

tA = t[A]
tB = t[B]
tC = t[C]
t['chemLabel'] = getChemLabel(tA, tB, tC)

#%%

# triangularPoints = list(  t.apply( lambda t: [ t['Xt'], t['Yt'], t['E/meV']/1000]  )   )

nXt = len(t['Xt'])
triangularPoints = np.zeros((nXt, 3))
triangularPoints[:, 0] = t['Xt']
triangularPoints[:, 1] = t['Yt']
triangularPoints[:, 2] = t['E/meV'] / 1000

# XYE = list(  t.apply( lambda t: [ t['X'],  t['Y'],  t['E/meV'] ] )   )
X   =  list( t['X'] )
Y   =  list( t['Y'] )
E   =  list( t['E/meV']/1000 )
Xt  =  list( t['Xt'] )
Yt  =  list( t['Yt'] )
labels =  list( t['chemLabel'] )

# another method using pandas:
# df = pd.DataFrame({'tri':    t['tri']  })
# triangularPoints = df['tri'].tolist()

import scipy
from scipy.spatial import ConvexHull
hull = ConvexHull( triangularPoints ) # triangularPoints = (x,y,Energy)



#%%
bottomSimplices = scr.getBottomSimplices(X, Y, E, hull.simplices)


t[ t['composition'] == 235 ].sort_values(by=['E/meV'])


def differenceToConvexHullSurface(Xi, Yi, EmeV):
    eS = [ scr.getEcrossingHull(Xi, Yi, X, Y, E*1000, hull.simplices) ]
    return (EmeV/1000 - eS[0])*1000

differenceToConvexHullSurface(20, 30, -172.8811)


#%%

# t[t['composition']==2].sort('totalEnergy')
#
#
# eA = -6.8388
# eB = -1.8955
# eC = -7.5786/2
# eC
# eC2 = -3.7889
#
# eF = 7*eB + 1*eC
# eF
# 1000.0 *(-25.44 - eF)/8
#
# -25.44/8
#
#
t[ t['E/meV'] < -1000.0 ]

#%%
################################################################################
# Plot
################################################################################
# import TernaryHull.PlotlyScripts as pls
import PlotlyScripts as pls

Emin = -0.5
Emax = 0.2
limits = [0,100, 0,100, -1,1, Emin, Emax]
colorBarTitle = 'relaxedDFT+prot'
nameHTMLfile = 'relaxedDFT+prot.html'

pls.make3DPlotlyDouble(Xt, Yt, E, labels,bottomSimplices,\
                   limits, colorBarTitle, nameHTMLfile)

#%%
################################################################################
# PDF Plot
################################################################################
import Tex as tex
pdfName = "CoLiSn_DFTrelaxedwithProt"
# pdfName = "/Users/chinchay/Documents/2_codes/CoLiSn_withPrototypes/convexHullOfMTP/CoLiSn"
tex.getPDFconvexHull(Xt, Yt, E, labels, bottomSimplices, A, B, C, pdfName)

#%%

t[ t['composition'] == 306 ].sort('E/meV')

t[ t['composition'] == 100 ].sort('E/meV')

t[ t['composition'] == 10 ].sort('E/meV')

t[ t['composition'] == 20 ].sort('E/meV')


# t[ t['composition'] == 20 ][]

# t[ t['composition'] == 1 ].sort('E/meV')



t[ t['composition'] == 303 ].sort('E/meV')
t[ t['composition'] == 505 ].sort('E/meV')








t2 = t
################################################################################
# Reducing to the ones close to convex hull:
################################################################################

## Getting the convex hull surface:
dEmax = 0.064 # --> 7060 cfgs !
EcrossingHull = [ scr.getEcrossingHull(X[i],Y[i], X,Y,E, hull.simplices) for i in range(len(X)) ]


t2.add_column( graphlab.SArray(EcrossingHull), name='EcrossingHull')

t2['E/eV']        = t2['E/meV'] / 1000.0 # returning to eV units for "h"
t2['dE=E-Ecross'] = t2['E/eV'] - t2['EcrossingHull']
t2['isCloseHull'] = t2['dE=E-Ecross'] < dEmax
# h = h.add_row_number(column_name='id', start=0) # python likes to begin in zero
# h['chemLabel'] = t['chemLabel']
t2

t2 = t2[ t2['isCloseHull'] == 1 ]
t2.num_rows()


################################################################################
# Removing repeated structures
################################################################################
import pandas as pd
import numpy as np
def getDataframeFromSFrame(sf):
    df = pd.DataFrame({'id':          sf['id'],
                       'conf_id':     sf['conf_id'],
                       'composition': sf['composition'],
                       'E/meV':       np.round(sf['E/meV'], decimals=0),
                       'volume':      np.round(sf['volume'], decimals=0),
                       'mindist':     sf['mindist'],
                       'minSolid/Pi': np.round(sf['minSolid/Pi'], decimals=1),
                       '|a1|':        sf['|a1|'],
                       '|a2|':        sf['|a2|'],
                       '|a3|':        sf['|a3|'] })
    df = np.round(df, decimals=2)
    return df
##


def removeDuplicates(df):
    # this will remove 'repeated' rows considering columns named according to 'subset'
    df = df.drop_duplicates(\
                       subset=['composition', 'E/meV', 'minSolid/Pi',\
                               'volume', '|a1|', '|a2|', '|a3|'  ],\
                       keep='first')
    return df
##
# def savingToFile(fileOut):
#     graphlab.SFrame.save('data/training_data.csv', format='csv')


df = getDataframeFromSFrame(t2)
df = removeDuplicates(df)
df



v = list(df['conf_id'])
len(v)

t2['isUnrepeated'] = t2.apply( lambda t2: t2['conf_id'] in v )
t2


t2 = t2[ t2['isUnrepeated'] == 1 ]
t2.num_rows()
t2



t2[ t2['composition'] == 21 ].sort('E/meV')

t2[ t2['composition'] == 121 ].sort('E/meV')

# `t2` contains cfg close to 4*RMS above the convex hull, and are unrepeated ones.
# this served to reduce memory! now...
################################################################################
# ... Choosing the lowest 3 cfgs !
################################################################################

# further reducing memory:
t2.remove_column('mindist')
t2.remove_column('|a1|')
t2.remove_column('|a2|')
t2.remove_column('|a3|')
t2.remove_column('volume')
t2.remove_column('minSolid/Pi')
t2.remove_column('isCloseHull')
t2.remove_column('isUnrepeated')

t2.remove_column('E/eV')
t2.remove_column('nAtoms')

t2.num_rows()

t2['composition'][4]



newConf_id = []
for composition in range(1, t2.num_rows() ):
    g = t2[ t2['composition'] == composition ].sort('E/meV')
    for i in range( min(3, g.num_rows()) ):
        newConf_id.append( g['conf_id'][i] )


#
len(newConf_id)

# h.save('filtered.csv', format='csv')
f = open("/Users/chinchay/Documents/2_codes/CoLiSn/convexHull/3cfgs.txt", "w")
for i in range(len(newConf_id)):
    f.write( newConf_id[i] + "\n" )
#
f.close()



t2 = t
t2['filtered'] = t.apply( lambda t: t['conf_id'] in newConf_id )
t2 = t2[ t2['filtered'] == 1 ]
t2

t2.num_rows()



################################################################################
# ... Plotting lowest 3 cfgs !
################################################################################
# import TernaryHull.PlotlyScripts as pls

Emin = -0.5
Emax = 0.2
limits = [0,100, 0,100, -1,1, Emin, Emax]
# colorBarTitle = 'test4RMSE'
# nameHTMLfile = 'test4RMSE.html'

colorBarTitle = '3cfgs'
nameHTMLfile = '/Users/chinchay/Documents/2_codes/CoLiSn/convexHull/3cfgs.html'


bottomSimplicesfile = scr.getBottomSimplices(X, Y, E, hull.simplices)

n = len(t2)
X_dE = [ t2['X'][i] for i in range(n) ]
Y_dE = [ t2['Y'][i] for i in range(n) ]
E_dE = [ t2['E/eV'][i] for i in range(n) ]

# converting from cartesians to Triangulars
Xt_dE = [ scr.getXt_fromCart(X_dE[i], Y_dE[i]) for  i in range(n) ]
Yt_dE = [ scr.getYt_fromCart(X_dE[i], Y_dE[i]) for  i in range(n) ]

# pls.make3DPlotly(Xt_dE, Yt_dE, E_dE, Xt, Yt, E, labels, bottomSimplicesfile,\
#                        limits, colorBarTitle,\
#                        nameHTMLfile) #boxCorners = [xmin,xmax, ymin,ymax, zmin, zmax]


f = open("/Users/chinchay/Documents/2_codes/CoLiSn/convexHull/3cfgs.txt", "w")
for i in range(len(t2)):
    f.write( t2['conf_id'][i] + "   " + t2['chemLabel'][i] + "\n" )
#
f.close()

# %%
