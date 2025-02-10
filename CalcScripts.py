import numpy as np

######################################################################  
# simplices = hull.simplices
def getOrderedIndicesOfConvexHull(simplices):
    orderedIndices = []
    for s in simplices:
        orderedIndices.append( s[0] ) #index of firstVertex of simplex `s`
        orderedIndices.append( s[1] ) #index of secondVertex
        orderedIndices.append( s[2] ) #index of thirdVertex
    
    # "set" will do UNION (avoiding repetition),
    # and "list" will convert the result {} into a list [].
    orderedIndices = sorted( list( set( orderedIndices ) ) )
    return orderedIndices

######################################
def getLoopedSimplicesInConvexHull(simplices):
    loopedSimplices = []
    for s in simplices:
        loopedSimplices.append( [ s[0], s[1], s[2], s[0] ] ) # Loop over the indices of the simplex `s`
    return loopedSimplices

######################################
def getOrderedData(X, simplices):
    orderedIndices = getOrderedIndicesOfConvexHull(simplices)
    return [ X[i] for i in orderedIndices ]

######################################
def getLoopedData(X, simplices):     
    loopedSimplices = getLoopedSimplicesInConvexHull(simplices)
    loopedX = []
    for i in range( len(loopedSimplices) ):  
        x0 = X[ loopedSimplices[i][0] ]
        x1 = X[ loopedSimplices[i][1] ]
        x2 = X[ loopedSimplices[i][2] ]
        x3 = X[ loopedSimplices[i][3] ]
        loopedX.append( [ x0, x1, x2, x3 ] )
    #
    return loopedX

#####################################
def getBottomSimplices(X, Y, E, simplices):
    bottomSimplices = []
    for s in simplices:
        barycenter = getBarycenterOfCartesianSimplex(X, Y, E, s)

        # it would need to improve the idea here, since I just take the energy of any vertix simplex
        energyTop = max([E[ s[0] ], E[ s[1] ], E[ s[2] ]])
        # energyTop = E[ s[0] ]
        
        if (    (not isBarycenterAtCartesianWalls(barycenter)) \
            and (not isEnergyAbove(energyTop))   ):
            bottomSimplices.append(s)
    #
    # check if a simplice is over other one. Eliminate the top one.
    n = len(bottomSimplices)
    correctBottomS = []
    for i in range(n):
        s = bottomSimplices[i]





    return bottomSimplices



#########################################################################################
#########################################################################################
def getBarycenter(r1, r2, r3):
    # r1, r2, and r3 are vectors, then the position of the barycenter is:
    x = ( r1[0] + r2[0] + r3[0] ) / 3.0
    y = ( r1[1] + r2[1] + r3[1] ) / 3.0
    z = ( r1[2] + r2[2] + r3[2] ) / 3.0
    barycenter = [x, y, z]
    return barycenter

def belongs(x, xmin, xmax):
    return (xmin <= x) and (x <= xmax)


def hypotenuseWall(x,lenghtLegA, lenghtLegB): ## would depend of the length of the legs (catetos)
    #  |_\  : legA, legB, hypotenuse
    # (x,y)/(x0,y0) = (-legA/legB), where (x0,y0)=(legB,0)
    y = (-lenghtLegA / lenghtLegB) * (x - lenghtLegB)
    return y 
    
def isBarycenterAtCartesianWalls(barycenter):
    x = barycenter[0]
    y = barycenter[1]
    
    eps = 0.1
    lengthLeg = 100

    isSimplexInXwall = belongs(x, -eps, eps)
    isSimplexInYwall = belongs(y, -eps, eps)
    
    yHypotenuse = hypotenuseWall(x, lengthLeg, lengthLeg)
    isSimplexInHyponeuseWall = belongs(y, yHypotenuse - eps, yHypotenuse + eps)

    # if simplex belongs to X-wall, but do not touches the Y-wall:
    # if simplex belongs to Y-wall, but do not touches the X-wall:
    # if simplex belongs to "hypotenuse", but do not touches the X-wall nor the Y-wall:        
#     if (    ( isSimplexInXwall and !isSimplexInYwall and !isSimplexInHyponeuseWall )    \
#         or  ( isSimplexInYwall and !isSimplexInXwall and !isSimplexInHyponeuseWall )    \
#         or  ( isSimplexInHyponeuseWall and  !isSimplexInXwal and !isSimplexInYwal  )  ):
#         return true

    if ( (         isSimplexInXwall   \
          and (not isSimplexInYwall)  \
          and (not isSimplexInHyponeuseWall) )  \
       or \
         (         isSimplexInYwall   \
          and (not isSimplexInXwall)  \
          and (not isSimplexInHyponeuseWall) )  \
       or \
         (         isSimplexInHyponeuseWall     \
          and (not isSimplexInXwall)   \
          and (not isSimplexInYwall) ) \
       ):
        return True
    
    return False
 
def isEnergyAbove(energy):
    # it would need improvement of the idea here...
    reference = 0.012 #0.012 #0.012 #0.02
    eps = 0.010
    emin = reference - eps
    emax = reference + 10
    if ( belongs(energy, emin, emax) ):
        return True
    return False


#########################################################################################
#########################################################################################
def getVerticesFromSimplex2(X,Y,E, simplex):
	r1 = [ X[simplex[0]], Y[simplex[0]], E[simplex[0]] ]
	r2 = [ X[simplex[1]], Y[simplex[1]], E[simplex[1]] ]
	r3 = [ X[simplex[2]], Y[simplex[2]], E[simplex[2]] ]
	return r1, r2, r3

# a simplex is a set of 3 integers, each one associate a one point p.
def getBarycenterOfCartesianSimplex(X,Y,E, simplex):
	[r1, r2, r3] = getVerticesFromSimplex2(X,Y,E, simplex)
	barycenter = getBarycenter(r1, r2, r3) # barycenter is a vector [x,y,z]
	return barycenter



#########################################################################################
#########################################################################################


import math
def norma(v):
    return ( ( v[0] * v[0] )  +  ( v[1] * v[1] ) ) ** 0.5

def dot(v, w):    
    return ( v[0] * w[0] ) + ( v[1] * w[1] )

def determinant(v, w): #will give us the sign of vectorial product (cross product) in 2D:
    return ( v[0] * w[1] ) - ( v[1] * w[0] ) #if >0:counterclockwise, <0:clockwise

def areEqualIn2D(v, w):
    eps = 0.001
    difference = [v[0] - w[0], v[1] - w[1]]
    rhoDiff = norma(difference)
    return belongs(rhoDiff, -eps, eps)

def areCounterClockWise(v,w):
    if ( determinant(v,w) > 0 ):  ##if >0:counterclockwise, <0:clockwise
        return True
    return False # if the det < 0 then v is clockwise of w

def isLEthanPi(angle):
    return ( angle <= math.pi + 0.01) # <<<<<@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@!!!!!!!!! <<<<<<<<<<<<<<??????
    
def getAngleCounterClockwise(v1, v2, vc): # v's are 3D vectors, but only XY-components will be considered
#https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
    vA = [v1[0] - vc[0], v1[1] - vc[1]]
    vB = [v2[0] - vc[0], v2[1] - vc[1]]
    cosAngle = dot(vA, vB) / (norma(vA) * norma(vB))
    
    if (abs(cosAngle) > 1.0):
        dcosAngle = abs(cosAngle) - 1.0
        if (dcosAngle > 0.001):
            print("there is something wrong here.")
        else:
            cosAngle = np.sign(cosAngle) * 1.0
#         print("new approximation for cosAngle = ", cosAngle) 
#         print(" ")
    
    innerAngle = math.acos(cosAngle) # in radians

    if (areCounterClockWise(vA,vB)):
        return innerAngle
    else: 
        return  (2.0 * math.pi) - innerAngle


def isInsideProjectedXYTriangle(v1, v2, v3, vc): #v's are 3D vectors
    if ( areEqualIn2D(v1,vc) or areEqualIn2D(v2,vc) or areEqualIn2D(v3,vc) ):
        return True # vc is one point belonging to the convex Hull: v1,v2, or v3
    else:
        alpha = getAngleCounterClockwise(v1, v2, vc) # calculated on the xy-projected vectors
        beta  = getAngleCounterClockwise(v2, v3, vc) # calculated on the xy-projected vectors
        gamma = getAngleCounterClockwise(v3, v1, vc) # calculated on the xy-projected vectors        
        if ( isLEthanPi(alpha) and isLEthanPi(beta) and isLEthanPi(gamma) ):
            return True
        else:
            return False  
    
def swap(r,s):
    temp = r
    r = s
    s = temp
    return r,s

def getVectorsFromBarycenterToVertices(p1, p2, p3):
    barycenter = getBarycenter(p1, p2, p3)
    A = [p1[0] - barycenter[0], p1[1] - barycenter[1], p1[2] - barycenter[2]]
    B = [p2[0] - barycenter[0], p2[1] - barycenter[1], p2[2] - barycenter[2]]
    C = [p3[0] - barycenter[0], p3[1] - barycenter[1], p3[2] - barycenter[2]]
    return A, B, C

def isTriangleXYCounterClockWiseFromBarycenter(p1,p2,p3):
    A, B, C = getVectorsFromBarycenterToVertices(p1, p2, p3)
    if (areCounterClockWise(A,B)):
        if (areCounterClockWise(B,C)):
            return True
    return False

def makeTriangleXYCounterClockWise(p1, p2, p3):
    if (not isTriangleXYCounterClockWiseFromBarycenter(p1, p2, p3)):
        p1, p2 = swap(p1, p2)
    return p1, p2, p3
    

def solve2D(u1, u2, v):
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.solve.html
    #Solve the system of equations 3*t1 +   t2 = 9
    #                          and   t1 + 2*t2 = 8:
    # a = np.array([[3,1], [1,2]])
    # b = np.array([9,8])
    # x = np.linalg.solve(a, b)
    #
    # Solving in 2D: u1*t1 + u2*t2 = v
    # where u1, u2, and v are known
    [m, n] = u1
    [p, q] = u2
    [r, s] = v

#     print("u1,u2,v=", u1,u2,v)
    
    a = np.array([[m,p], [n,q]])
    b = np.array([r,s])
    x = np.linalg.solve(a, b) ## x is a 2D vector!
    [t1, t2] = x
    return t1, t2

def getUnitVector(v):
    normaV = norma(v)
    u = [ v[0] / normaV, v[1] / normaV, v[2] / normaV ]
    return u

def getXYprojectedVector(v):
    vXY = [v[0], v[1]] # v has x,y,z components!
    return vXY

# vStr = [x,y,Energy] of a configuration, in cartesian coordinates.
def getEnergyCrossingHullSurface(v1, v2, v3, vConfig):
#     A = v2 - v1
#     B = v3 - v1
#     print(v1, v2, v3)
    A = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]]
    B = [v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]]    
    [x, y, zTemp] = vConfig # zTemp is not going to be used here.
    
    # this function finds Eh in the vectorial equation: 
    # [x,y,Eh] = v1 + unitVector(A)*t1 + unitVector(B)*t2; t1,t2 are Reals
    # First, we find t1 and t2 by solving a system of two equations
    # embedded in the XY-components of the vectorial equation:
    # [x,y] = v1 + u2D_A*t1 + u2D_B*t2, to find t1,t2,
    # but we can reduce by doing:
    rxy = [x - v1[0], y - v1[1]]
    # So now, we need to solve rxy = u2D_A*t1 + u2D_B*t2, to find t1,t2
    
    # building the unit vectors:
    uA = getUnitVector(A)
    uB = getUnitVector(B)
    
    # proyecting into the XY-plane:
    u2D_A = getXYprojectedVector(uA)
    u2D_B = getXYprojectedVector(uB)
    
    # finding t1,t2 that solves rxy = u2D_A*t1 + u2D_B*t2  
    t1, t2 = solve2D(u2D_A, u2D_B, rxy)
    
    # now we can find Eh in the original vector 3D equation:
    # [x,y,Eh] = v1 + unitVector(A)*t1 + unitVector(B)*t2; t1,t2 are Reals
    Eh = v1[2] + (uA[2] * t1) + (uB[2] * t2)
    return Eh
    


#########################################################################################
#########################################################################################
def convertCartesianToTriang(x,y):
    import math # importing "math" for mathematical operations
    x0 = x
    y0 = y
    x = x0 + (0.5 * y0)
    y = y0 * (math.sqrt(3.0) / 2.0)
    return x,y


#########################################################################################
#########################################################################################
def getCartesianUniformPoints(partitions):
    xu = []
    yu = []
    zu = []
    for iA in range(0, partitions + 1):
        for iC in range(0, partitions + 1):
            x = iA * 100.0 / partitions
            z = iC * 100.0 / partitions
            y = (100 - z) - x
            if (y >= 0.0):
                xu.append(x)
                yu.append(y)
                zu.append(0.0)
    return xu, yu, zu

def getEcrossingHull(xTest, yTest, X,Y,E, simplices):
    e = 0.0
    bottomSimplices = getBottomSimplices(X,Y,E, simplices)
    for s in bottomSimplices:
        r1, r2, r3 = getVerticesFromSimplex2(X, Y, E, s) # cartesian!
        v1, v2, v3 = makeTriangleXYCounterClockWise(r1, r2, r3) # only accepts cartesian, return cartesian!
        rConfig = [xTest, yTest, e] # e=0.0 will not be used

        if ( isInsideProjectedXYTriangle(v1, v2, v3, rConfig) ): # cartesian compared to cartesians!
            EcrossingHull = getEnergyCrossingHullSurface(v1, v2, v3, rConfig)
            return EcrossingHull
    #
    return -10000


#########################################################################################
#########################################################################################
def getCartesianUniformPoints(partitions):
    xu = []
    yu = []
    zu = []
    for iA in range(0, partitions + 1):
        for iC in range(0, partitions + 1):
            x = iA * 100.0 / partitions
            z = iC * 100.0 / partitions
            y = (100 - z) - x
            if (y >= 0.0):
                xu.append(x)
                yu.append(y)
                zu.append(0.0)
    return xu, yu, zu

def getEcrossingHull(xTest, yTest, X,Y,E, simplices):
    e = 0.0
    bottomSimplices = getBottomSimplices(X,Y,E, simplices)
    for s in bottomSimplices:
        r1, r2, r3 = getVerticesFromSimplex2(X, Y, E, s) # cartesian!
        v1, v2, v3 = makeTriangleXYCounterClockWise(r1, r2, r3) # only accepts cartesian, return cartesian!
        rConfig = [xTest, yTest, e] # e=0.0 will not be used

        if ( isInsideProjectedXYTriangle(v1, v2, v3, rConfig) ): # cartesian compared to cartesians!
            EcrossingHull = getEnergyCrossingHullSurface(v1, v2, v3, rConfig)
            return EcrossingHull
    #
    return -10000

def getZsurfaceHull(X,Y,E, simplices, partitions):
    # partitions = 80
    x_grid, y_grid, temp = getCartesianUniformPoints(partitions)
    ecrossingHull = []
    for i in range(len(x_grid)):
        x = x_grid[i]
        y = y_grid[i]    
        z = getEcrossingHull(x,y, X,Y,E, simplices) # internally will take just the bottomSimplices
        ecrossingHull.append(z)
    #
    return x_grid, y_grid, ecrossingHull


def getDataFromFile(fileName, A, B, C):
    import sys
    sys.path.append('/Users/chinchay/Documents/8_librerias/xPythonPackages/')
    import graphlab
    from scipy.spatial import ConvexHull
    import TernaryHull # located in /Users/chinchay/Documents/8_librerias/xPythonPackages/
#     import TernaryHull.CalcScripts as scr
#     import TernaryHull.PlotlyScripts as pls
    # Limit number of worker processes. This preserves system memory, which prevents
    # hosted notebooks from crashing.
    graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)    

    obj = TernaryHull.GraphlabClass(fileName, A, B, C)
    cart, triang, labels = obj.get_Cart_Triang_labels()

    triangularPoints = [ [triang[i][0], triang[i][1], triang[i][2]] for i in range(len(triang)) ]
    cartesianPoints  = [ [cart[i][0], cart[i][1], cart[i][2]] for i in range(len(cart)) ]

    X  = [ cartesianPoints[i][0] for i in range(len(cartesianPoints)) ]
    Y  = [ cartesianPoints[i][1] for i in range(len(cartesianPoints)) ]
    Xt = [ triang[i][0] for i in range(len(triang)) ]
    Yt = [ triang[i][1] for i in range(len(triang)) ]
    E  = [ triang[i][2] for i in range(len(triang)) ]
    labels = [ labels[i] for i in range(len(cart)) ]
    hull = ConvexHull( triangularPoints ) # triangularPoints = (x,y,Energy)
    table = obj.getTable()

    return X, Y, Xt, Yt, E, labels, hull, table, obj # obj=tableGraphClass
#########

def getXt_fromCart(x, y):
    import math # importing "math" for mathematical operations
    xt = x + (0.5 * y)
#     y = y0 * (math.sqrt(3.0) / 2.0)
    return xt

def getYt_fromCart(x, y):
    import math # importing "math" for mathematical operations
#     xt = x + (0.5 * y)
    yt = y * (math.sqrt(3.0) / 2.0)
    return yt

#########

def areComponentsLessEqual9(t, A, B, C):
    import graphlab
    # A='Co', B='Ni', C='Ti'
    isA_LE_9 = graphlab.SArray( [ t[A][i] <= 9 for i in range(len(t))   ]   ).all()
    isB_LE_9 = graphlab.SArray( [ t[B][i] <= 9 for i in range(len(t))   ]   ).all()
    isC_LE_9 = graphlab.SArray( [ t[C][i] <= 9 for i in range(len(t))   ]   ).all()
    return isA_LE_9 & isB_LE_9 & isC_LE_9

def getTypeComposition(t, A, B, C):
    t['composition'] = (t[A] * 100) + (t[B] * 10) + (t[C] * 1)
    
    # this part takes a lot of time since we use areComponentsLessEqual9 which is a FOR loop !!!    
    # if ( areComponentsLessEqual9(t, A, B, C) ): # A='Co', B='Ni', C='Ti'
    #     # THE FOLLOWING IS ONLY VALID IF nCo <=9, nNi <= 9, nTi <=9 <<<<<<<<<<<<<<<<<<@@@@@@@@@@@@@@@@@
    #     t['composition'] = (t[A] * 100) + (t[B] * 10) + (t[C] * 1)
    # else:
    #     print('More than 9 atoms!!! You should use a numerical base greater than 10.')
    # #
    return t


# # function `norm` already exists!
# # this one works for columns to avoid for loops (to work with graplab)
# # also, this works for 3D: x,y,z
# def norma2(x, y, z):
#     return ( ( x * x )  +  ( y * y ) + ( z * z ) ) ** 0.5



