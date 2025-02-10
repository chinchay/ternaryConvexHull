def dot2(Vx, Vy, Vz, Wx, Wy, Wz):
    return (Vx * Wx) + (Vy * Wy) + (Vz * Wz)
#

def cross(Vx, Vy, Vz, Wx, Wy, Wz):
    Mx = -Vz * Wy  +  Vy * Wz
    My =  Vz * Wx  -  Vx * Wz
    Mz = -Vy * Wx  +  Vx * Wy
    return Mx, My, Mz
#

def getVolume(t):
    """"
    A1 = {a1x, a1y, a1z};
    A2 = {a2x, a2y, a2z};
    A3 = {a3x, a3y, a3z};
    FortranForm[ Dot[ Cross[A1, A2], A3  ] ] =
    (-(a1z*a2y) + a1y*a2z)*a3x +\
      (a1z*a2x  - a1x*a2z)*a3y +\
    (-(a1y*a2x) + a1x*a2y)*a3z
    We use abs() to avoid negative values of volume due to A1,A2,A3 being counterclockwise
    """
    return abs(
                     ( -(t['a1z']*t['a2y']) + t['a1y']*t['a2z'] ) * t['a3x']  +\
                     (   t['a1z']*t['a2x']  - t['a1x']*t['a2z'] ) * t['a3y']  +\
                     ( -(t['a1y']*t['a2x']) + t['a1x']*t['a2y'] ) * t['a3z']

                    )

##############################################################################
def getNorm3D(v):
    return ( ( v[0] * v[0] )  +  ( v[1] * v[1] ) + ( v[2] * v[2] ) ) ** 0.5


def getEnergy(fileName):
    energies = []
    with open(fileName, "r") as ifile:
        for line in ifile:
            if "Energy" in line:
                energies.append( float( ifile.next() ) )
    #
    return energies

def getLatticeVector(fileName):
    a1xL = []; a1yL = []; a1zL = []
    a2xL = []; a2yL = []; a2zL = []
    a3xL = []; a3yL = []; a3zL = []

    with open(fileName, "r") as ifile:
        for line in ifile:
            if "Supercell" in line:
                line1 = ifile.next()
                line2 = ifile.next()
                line3 = ifile.next()
#                 a1x, a1y, a1z = [ float( line1.split()[i] ) for i in range(3) ]
#                 a2x, a2y, a2z = [ float( line2.split()[i] ) for i in range(3) ]
#                 a3x, a3y, a3z = [ float( line3.split()[i] ) for i in range(3) ]
                a1 = [ float( line1.split()[i] ) for i in range(3) ]
                a2 = [ float( line2.split()[i] ) for i in range(3) ]
                a3 = [ float( line3.split()[i] ) for i in range(3) ]

                a1, a2, a3 = sorted( [a1, a2, a3], key=getNorm3D )

                a1x, a1y, a1z = a1
                a2x, a2y, a2z = a2
                a3x, a3y, a3z = a3

                a1xL.append( a1x )
                a1yL.append( a1y )
                a1zL.append( a1z )

                a2xL.append( a2x )
                a2yL.append( a2y )
                a2zL.append( a2z )

                a3xL.append( a3x )
                a3yL.append( a3y )
                a3zL.append( a3z )

    #
    return a1xL, a1yL, a1zL, a2xL, a2yL, a2zL, a3xL, a3yL, a3zL


def getMindist(fileName):
    """ First do on the terminal
    mlp mindist relaxed.cfg"
    """
    mindistL = []
    with open(fileName, "r") as ifile:
        for line in ifile:
            if "Feature   mindist" in line:
                mindistL.append( float( line.split()[2] ) )
    #
    return mindistL

def getConfid(fileName):
    """ First do on the terminal
    mlp mindist relaxed.cfg"
    """
    conf_id = []
    with open(fileName, "r") as ifile:
        for line in ifile:
            if "Feature   conf_id" in line:
                conf_id.append( line.split()[2] )
    #
    return conf_id

def getComposition(fileName):
    nAlist = []
    nBlist = []
    nClist = []

    with open(fileName, "r") as ifile:
        for line in ifile:
            if "Size" in line:
                size = int( ifile.next() )

                temp = ifile.next() # "Supercell"
                temp = ifile.next() # a1
                temp = ifile.next() # a2
                temp = ifile.next() # a3
                temp = ifile.next() # "AtomData"

                nA = 0
                nB = 0
                nC = 0
                for i in range(size):
                    v = ifile.next()
                    typeAtom = int( v.split()[1] )

                    if (typeAtom == 0):
                        nA += 1
                    elif (typeAtom == 1):
                        nB += 1
                    elif (typeAtom == 2):
                        nC += 1
                #
                assert size == nA + nB + nC
                #
                nAlist.append(nA)
                nBlist.append(nB)
                nClist.append(nC)
    #
    return nAlist, nBlist, nClist
#################################################################################

def getEnergyPerAtom(table, A, B, C):
    t = table
    t['nAtoms'] = t[ A ] + t[ B ] + t[ C ]
    t['EperAtom'] = t['totalEnergy'] / t['nAtoms']
    return t

def getPureA(table, A, B, C):
    t = table
    onlyCo = t[ t[ B ] == 0 ]
    onlyCo = onlyCo[ onlyCo[ C ] == 0 ]
    minEperAtom_CoPure = onlyCo[ onlyCo['EperAtom'] == onlyCo['EperAtom'].min() ]
    return minEperAtom_CoPure

def getPureB(table, A, B, C):
    t = table
    onlyNi = t[ t[ A ] == 0 ]
    onlyNi = onlyNi[ onlyNi[ C ] == 0 ]
    minEperAtom_NiPure = onlyNi[ onlyNi['EperAtom'] == onlyNi['EperAtom'].min() ]
    return minEperAtom_NiPure

def getPureC(table, A, B, C):
    t = table
    onlyTi = t[ t[ A ] == 0 ]
    onlyTi = onlyTi[ onlyTi[ B ] == 0 ]
    minEperAtom_TiPure = onlyTi[ onlyTi['EperAtom'] == onlyTi['EperAtom'].min() ]
    return minEperAtom_TiPure

def getFormationEnergy(table, A, B, C):
    t = table
    t = getEnergyPerAtom(t, A, B, C)

    minEperAtom_A_pure = getPureA(table, A, B, C)
    minEperAtom_B_pure = getPureB(table, A, B, C)
    minEperAtom_C_pure = getPureC(table, A, B, C)

    ##############################
    # ADDING SOME DEFAULT VALUES FOR COBALT, NIQUEL, AND TITATIUM:
    ##############################
    print(minEperAtom_A_pure['EperAtom'])

    eA = minEperAtom_A_pure['EperAtom'].iloc[0] #if len(minEperAtom_A_pure) > 0 else -14.10392 / 2
    eB = minEperAtom_B_pure['EperAtom'].iloc[0] #if len(minEperAtom_B_pure) > 0 else -21.99672 / 4
    eC = minEperAtom_C_pure['EperAtom'].iloc[0] #if len(minEperAtom_C_pure) > 0 else -15.56099 / 2

    t['EbeforeReaction'] =  t[A] * eA +\
                            t[B] * eB +\
                            t[C] * eC

    t['E/meV'] = 1000.0 * (t['totalEnergy'] - t['EbeforeReaction']) / t['nAtoms']

    # removing unnecessary columns:
    # t = t.drop('EbeforeReaction')
    t = t.drop(['EperAtom'], axis=1)

    return t


################################################################################

def dot(A, B): # A,B,C are vectors [x,y,z]
    return A[0]*B[0] + A[1]*B[1] + A[2]*B[2]

def cross(A,B): # A,B,C are vectors [x,y,z]
    x = -A[2] * B[1]  +  A[1] * B[2]
    y =  A[2] * B[0]  -  A[0] * B[2]
    z = -A[1] * B[0]  +  A[0] * B[1]
    return [x, y, z]

def getVol(A,B,C): # A,B,C are vectors [x,y,z]
    return abs( dot( cross(A, B) , C ) )

def getSolidAngle(A,B,C): # A,B,C are vectors [x,y,z]
    import math
    pi = math.pi

    a = getNorm3D(A)
    b = getNorm3D(B)
    c = getNorm3D(C)

    num = getVol(A,B,C)
    denom1 = a * b * c
    denom2 = dot(A, B) * c
    denom3 = dot(A, C) * b
    denom4 = dot(B, C) * a

    # tan(omega/2) = num/denom https://en.wikipedia.org/wiki/Solid_angle
    tan = num / (denom1 + denom2 + denom3 + denom4)
    omega = 2.0 * math.atan(tan)
    omega = omega if omega >= 0.0 else omega + (2.0 * pi)

    return omega

def getMinSolidPi(A,B,C): # A,B,C are vectors [x,y,z]
    import math
    import TernaryHull.CalcScripts as scr

    om1 = getSolidAngle(A,B,C)

    As  = [-A[0], -A[1], -A[2]]
    Bs  = B
    Cs  = C
    om2 = getSolidAngle(As, Bs, Cs)

    As  = A
    Bs  = [-B[0], -B[1], -B[2]]
    Cs  = C
    om3 = getSolidAngle(As, Bs, Cs)

    As  = A
    Bs  = B
    Cs  = [-C[0], -C[1], -C[2]]
    om4 = getSolidAngle(As, Bs, Cs)

    minOmeg = min([om1,om2,om3,om4])
    pi2 = math.pi * 2.0
    sumOm = om1 + om2 + om3 + om4
    assert scr.belongs(sumOm, pi2 -0.1, pi2 + 0.1)
    return [minOmeg / math.pi, om1 + om2 + om3 + om4]


def getMinOmegas2(t):
    # import graphlab
    import math
    import numpy as np
    import pandas as pd
    pi = math.pi

    t['|a1|'] = normA1(t)
    t['|a2|'] = normA2(t)
    t['|a3|'] = normA3(t)
    t['volume'] = getVolume(t)

    t['d0'] = t['|a1|'] * t['|a2|'] * t['|a3|']

    ###################################################
    # 1,1,1
    t['d1'] = dot2(t['a1x'], t['a1y'], t['a1z'], t['a2x'], t['a2y'], t['a2z']) * t['|a3|']
    t['d2'] = dot2(t['a1x'], t['a1y'], t['a1z'], t['a3x'], t['a3y'], t['a3z']) * t['|a2|']
    t['d3'] = dot2(t['a2x'], t['a2y'], t['a2z'], t['a3x'], t['a3y'], t['a3z']) * t['|a1|']

    t['tan'] = t['volume'] / (t['d0'] + t['d1'] + t['d2'] + t['d3'])

    df = pd.DataFrame( {'tan': t['tan'] } )
    df['tan'] = np.arctan( df )
    sa = pd.DataFrame(df['tan'].values) #graphlab.SArray
    t['omtemp'] = sa
    t['omtemp'] = 2.0 * t['omtemp']
    # t['om1'] = t['omtemp'] if t['omtemp'] >= 0.0 else t['omtemp'] + (2.0 * pi)
    t['om1'] = t['omtemp'].apply(lambda x: x if x >= 0.0 else x + (2.0 * pi))

    t.drop(['d1', 'd2', 'd3', 'tan', 'omtemp'], axis=1)
    ###################################################
    # -1,1,1
    s1 = -1.0
    s2 =  1.0
    s3 =  1.0
    t['d1'] = dot2(s1*t['a1x'], s1*t['a1y'], s1*t['a1z'], s2*t['a2x'], s2*t['a2y'], s2*t['a2z']) * t['|a3|']
    t['d2'] = dot2(s1*t['a1x'], s1*t['a1y'], s1*t['a1z'], s3*t['a3x'], s3*t['a3y'], s3*t['a3z']) * t['|a2|']
    t['d3'] = dot2(s2*t['a2x'], s2*t['a2y'], s2*t['a2z'], s3*t['a3x'], s3*t['a3y'], s3*t['a3z']) * t['|a1|']

    t['tan'] = t['volume'] / (t['d0'] + t['d1'] + t['d2'] + t['d3'])

    df = pd.DataFrame( {'tan': t['tan'] } )
    df['tan'] = np.arctan( df )
    sa = pd.DataFrame(df['tan'].values) #graphlab.SArray
    t['omtemp'] = sa
    t['omtemp'] = 2.0 * t['omtemp']
    # t['om2'] = t['omtemp'] if t['omtemp'] >= 0.0 else t['omtemp'] + (2.0 * pi)
    t['om2'] = t['omtemp'].apply(lambda x: x if x >= 0.0 else x + (2.0 * pi))

    t.drop(['d1', 'd2', 'd3', 'tan', 'omtemp'], axis=1)
    ###################################################
    # 1,-1,1
    s1 =  1.0
    s2 = -1.0
    s3 =  1.0
    t['d1'] = dot2(s1*t['a1x'], s1*t['a1y'], s1*t['a1z'], s2*t['a2x'], s2*t['a2y'], s2*t['a2z']) * t['|a3|']
    t['d2'] = dot2(s1*t['a1x'], s1*t['a1y'], s1*t['a1z'], s3*t['a3x'], s3*t['a3y'], s3*t['a3z']) * t['|a2|']
    t['d3'] = dot2(s2*t['a2x'], s2*t['a2y'], s2*t['a2z'], s3*t['a3x'], s3*t['a3y'], s3*t['a3z']) * t['|a1|']

    t['tan'] = t['volume'] / (t['d0'] + t['d1'] + t['d2'] + t['d3'])

    df = pd.DataFrame( {'tan': t['tan'] } )
    df['tan'] = np.arctan( df )
    sa = pd.DataFrame(df['tan'].values) #graphlab.SArray
    t['omtemp'] = sa
    t['omtemp'] = 2.0 * t['omtemp']
    # t['om3'] = t['omtemp'] if t['omtemp'] >= 0.0 else t['omtemp'] + (2.0 * pi)
    t['om3'] = t['omtemp'].apply(lambda x: x if x >= 0.0 else x + (2.0 * pi))

    t.drop(['d1', 'd2', 'd3', 'tan', 'omtemp'], axis=1)
    ###################################################
    # 1,1,-1
    s1 =  1.0
    s2 =  1.0
    s3 = -1.0
    t['d1'] = dot2(s1*t['a1x'], s1*t['a1y'], s1*t['a1z'], s2*t['a2x'], s2*t['a2y'], s2*t['a2z']) * t['|a3|']
    t['d2'] = dot2(s1*t['a1x'], s1*t['a1y'], s1*t['a1z'], s3*t['a3x'], s3*t['a3y'], s3*t['a3z']) * t['|a2|']
    t['d3'] = dot2(s2*t['a2x'], s2*t['a2y'], s2*t['a2z'], s3*t['a3x'], s3*t['a3y'], s3*t['a3z']) * t['|a1|']

    t['tan'] = t['volume'] / (t['d0'] + t['d1'] + t['d2'] + t['d3'])

    df = pd.DataFrame( {'tan': t['tan'] } )
    df['tan'] = np.arctan( df )
    sa = pd.DataFrame(df['tan'].values) #graphlab.SArray
    t['omtemp'] = sa
    t['omtemp'] = 2.0 * t['omtemp']
    # t['om4'] = t['omtemp'] if t['omtemp'] >= 0.0 else t['omtemp'] + (2.0 * pi)
    t['om4'] = t['omtemp'].apply(lambda x: x if x >= 0.0 else x + (2.0 * pi))

    t.drop(['d1', 'd2', 'd3', 'tan', 'omtemp'], axis=1)
    ###################################################
    t.drop(['d0'], axis=1)

    t['sumOm'] = t['om1'] + t['om2'] + t['om3'] + t['om4']

    import pandas as pd
    df = pd.DataFrame( {'om1': t['om1'],
                        'om2': t['om2'],
                        'om3': t['om3'],
                        'om4': t['om4'] } )

    df['min'] = df.min(axis=1)

    print("adding column...")
    # sa = pd.DataFrame( df['min'].values ) #graphlab.SArray
    print("adding column...")

    # t.add_column(sa, name='minSolid/Pi')
    t['minSolid/Pi'] = df['min'].values
    t['minSolid/Pi']  = t['minSolid/Pi'] / pi

    # falta trabajar en esto para columnas:
    # pi2 = math.pi * 2.0
    # assert scr.belongs(t['sumOm'], pi2 -0.1, pi2 + 0.1)
    print("finished.")

    return t




###############################################################################
def getMinOmegas(t):
    minOmegaPi = []
    # sumOmegasPi = []
    for i in range(len(t)):
        A1 = [ t['a1x'][i], t['a1y'][i], t['a1z'][i] ]
        A2 = [ t['a2x'][i], t['a2y'][i], t['a2z'][i] ]
        A3 = [ t['a3x'][i], t['a3y'][i], t['a3z'][i] ]
        minOm, sumOm = getMinSolidPi(A1, A2, A3)
        minOmegaPi.append( minOm )
    #     sumOmegasPi.append( sumOm )
    return minOmegaPi

##############################################################################

def normA1(t):
    return ( ( t['a1x'] * t['a1x'] )  +  ( t['a1y'] * t['a1y'] ) + ( t['a1z'] * t['a1z'] ) ) ** 0.5    
def normA2(t):
    return ( ( t['a2x'] * t['a2x'] )  +  ( t['a2y'] * t['a2y'] ) + ( t['a2z'] * t['a2z'] ) ) ** 0.5
def normA3(t):
    return ( ( t['a3x'] * t['a3x'] )  +  ( t['a3y'] * t['a3y'] ) + ( t['a3z'] * t['a3z'] ) ) ** 0.5




################################################################################
# FIRST, YOU SHOULD FIX THE LATTICE VECTORS...
# typing "mlp help convert-cfg" will get the list options of mlp
# one of them is useful to rearrange lattice vectors: --fix-lattice
# I have used "mlp convert-cfg --fix-lattice relaxed.cfg outFixedLatVect.cfg"
# also I have done: "mlp mindist relaxed.cfg" to include mindist in every configuration.
####################################################################################
def buildTable(conf_idL, nAlist, nBlist, nClist, energies, mindistL,\
                a1xL, a1yL, a1zL,\
                a2xL, a2yL, a2zL,\
                a3xL, a3yL, a3zL,\
                A, B, C ):
    import pandas as pd
    import CalcScripts as scr

    d = {
        "conf_id": conf_idL,
        A: nAlist,
        B: nBlist,
        C: nClist,
        "totalEnergy": energies,
        'a1x': a1xL,
        'a1y': a1yL,
        'a1z': a1zL,
        'a2x': a2xL,
        'a2y': a2yL,
        'a2z': a2zL,
        'a3x': a3xL,
        'a3y': a3yL,
        'a3z': a3zL,
        'mindist': mindistL,
    }

    t = pd.DataFrame(d)
    t = scr.getTypeComposition(t, A, B, C)
    # print("got composition")

    


    # t = t.add_row_number(column_name='id', start=0) # python likes to begin in zero
    # print("added id")
    t = getFormationEnergy(t, A, B, C)
    print("calculated formationEnergy")

    

    # t['volume'] = t.apply( getVolume() )
    # print("got volume")
    # t = t.add_columns(pd.DataFrame( {'mindist': mindistL} ))
    # print("got mindist")



    t = getMinOmegas2(t)

    

    # sa = graphlab.SArray(getMinOmegas(t))
    # print("got omegas")
    # t['minSolid/Pi'] = sa


    # t['|a1|'] = t.apply( normA1() )
    # print("got normA1")
    # t['|a2|'] = t.apply( normA2() )
    # print("got normA2")
    # t['|a3|'] = t.apply( normA3() )
    # print("got normA3")

    # t.drop( ['totalEnergy', 'a1x', 'a1y', 'a1z', 'a2x', 'a2y', 'a2z', 'a3x', 'a3y', 'a3z',\
    #                   'om1', 'om2', 'om3', 'om4', 'sumOm' ] )
    # t.drop( ['a1x', 'a1y', 'a1z', 'a2x', 'a2y', 'a2z', 'a3x', 'a3y', 'a3z',\
                #   'om1', 'om2', 'om3', 'om4', 'sumOm' ] )

    # names = ['totalEnergy', 'a1x', 'a1y', 'a1z', 'a2x', 'a2y', 'a2z', 'a3x', 'a3y', 'a3z',\
    #                   'om1', 'om2', 'om3', 'om4', 'sumOm' ]
    
    # t.drop(names, axis=1)

    

    return t
