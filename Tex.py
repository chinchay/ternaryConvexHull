# import TernaryHull.CalcScripts as scr
import CalcScripts as scr

# Note:
# What are interpolated are the color at each vertex
# Linear interpolation of colors work well with colormap of just 2 colors(linear), not 3(quadratic?)
# For just some points as in the case of stable structures in a convex hull use just 2 colors.


# def getChemTeX(string):
#     """ This function convert the string CoxNiyTiz into a TeX format: $Co_xNi_yTi_z$
#         We assume chemical formula begins with a letter!
#     """
#     lista = list(string)
#     chemTeX = "$"
#     previousWasDigit = False
#     for i in range( len(lista) ):
#         if( lista[i].isdigit() ):
#             if (previousWasDigit == False):
#                 chemTeX = chemTeX + "_{" + lista[i]
#             else:
#                 chemTeX = chemTeX + lista[i]
#             previousWasDigit = True
#         else:
#             if (previousWasDigit == True):
#                 chemTeX = chemTeX + "}"
#             chemTeX = chemTeX + lista[i]
#             previousWasDigit = False

#     if (previousWasDigit == True):
#         chemTeX = chemTeX + "}$"
#     else:
#         chemTeX = chemTeX + "$"
#     return chemTeX

#########################################################################################################
def getChemAngle(elementA, elementB, elementC, chemText):
    import re
    # text = 'Co2Ni3Ti4'

    isThereA = chemText.find(elementA)
    isThereB = chemText.find(elementB)
    isThereC = chemText.find(elementC)

    if ( (isThereA > -1) and (isThereB > -1) and (isThereC > -1)):
            angleDeg = 0
            xshift   = 0
            yshift   = 0
            color = "white"
    elif ( (isThereA > -1) and (isThereB > -1) ):
            angleDeg = -30
            xshift   = -5.5
            yshift   = 3.5
            color = "black"
    elif ( (isThereA > -1) and (isThereC > -1) ):
            angleDeg = 90
            xshift   = 0
            yshift   = -7
            color = "black"
    elif ( (isThereB > -1) and (isThereC > -1) ):
            angleDeg = 30
            xshift   = 5.5
            yshift   = 3.5
            color = "black"
    elif ( ( ( (isThereA > -1) ) and (  (isThereB == -1) or (isThereC == -1) ) ) or\
           ( ( (isThereC > -1) ) and (  (isThereA == -1) or (isThereB == -1) ) )  ):
            angleDeg = 90
            xshift   = 0
            yshift   = -5
            color = "black"
    elif ( ( (isThereB > -1) ) and (  (isThereA == -1) or (isThereC == -1) ) ):
            angleDeg = 0
            xshift   = 0
            yshift   = 4
            color = "black"

    return (color, xshift, yshift, angleDeg)
#

#########################################################################################################
def getXYZ(xList, yList, zList, iIndexes, i):
    #---------------------------------------------------------------------------
    # x,y,z are the triangular-energy coordinates of points belonging to the
    # a simplex of the convex hull:
    x  = xList[ iIndexes[i] ] * (-10)
    y  = yList[ iIndexes[i] ] * 10
    z  = zList[ iIndexes[i] ] * 1000
    return (x, y, z)

#########################################################################################################
def getStringFromXYZfloats(x, y, z):
#     print(z)
    xs = '{:f}'.format(x)
    ys = '{:f}'.format(y)
    zs = '{:f}'.format(z)
    return (xs, ys, zs)

#########################################################################################################
def getCircleTeX(x, y):
    xs, ys, zs = getStringFromXYZfloats(x, y, 0.0)
    string = "\draw[line width = 2pt] (axis cs: " + xs + "," + ys + ") circle (2pt);\n"
    return string

#########################################################################################################
def getLineTeX(x, y, x2, y2):
    xs,  ys,  zs  = getStringFromXYZfloats(x, y, 0.0)
    x2s, y2s, z2s = getStringFromXYZfloats(x2, y2, 0.0)
    string = "\draw[thick](axis cs: " + xs  + ", " + ys  +\
                     " )--(axis cs: " + x2s + ", " + y2s + " )node[]{};\n"
    return string

#########################################################################################################
def getLabelTeX(A, B, C, label, x,y):
    xs, ys, zs = getStringFromXYZfloats(x, y, 0.0)
    color, xshift, yshift, angleDeg = getChemAngle(A, B, C, label)
    string = "\\node[color= " + color  +\
              " ,xshift= " + repr(xshift)   +\
              " ,yshift= " + repr(yshift)   +\
              " ,rotate= " + repr(angleDeg) +\
              " ,scale=0.3] at (axis cs: " +\
              xs + ", " + ys + "){ \ch{ " + label + " } };\n"
    return string

def getLabelTeX2(A, B, C, label, x,y, scale):
    xs, ys, zs = getStringFromXYZfloats(x, y, 0.0)
    color, xshift, yshift, angleDeg = getChemAngle(A, B, C, label)
    string = "\\node[color= " + color  +\
              " ,xshift= " + repr(xshift)   +\
              " ,yshift= " + repr(yshift)   +\
              " ,rotate= " + repr(angleDeg) +\
              " ,scale= " + repr(scale) + " ] at (axis cs: " +\
              xs + ", " + ys + "){ \ch{ " + label + " } };\n"
    return string


#########################################################################################################
def getLineForFile(x, y, z):
    xs, ys, zs = getStringFromXYZfloats(x, y, z)
    string = xs + "   " + ys + "   " + zs + "\n"
    return string

def getBarycenter(x1,y1,z1, x2,y2,z2, x3,y3,z3):
    return [ (x1+x2+x3)/3.0, (y1+y2+y3)/3.0, (z1+z2+z3)/3.0]

def convertTriangToCartesian(x,y):
    import math # importing "math" for mathematical operations
    yc = y / (math.sqrt(3.0) / 2.0)
    xc = x - (0.5 * yc)
    return xc, yc

#########################################################################################################
def getSimplexTeX(x1,y1,z1, x2,y2,z2, x3,y3,z3):
    x1s, y1s, z1s = getStringFromXYZfloats(x1, y1, z1)
    x2s, y2s, z2s = getStringFromXYZfloats(x2, y2, z2)
    x3s, y3s, z3s = getStringFromXYZfloats(x3, y3, z3)
    space  =  "   "
    string =  "   \\addplot[patch,shader=interp, opacity=1.0]\n"         +\
              "            table[point meta=\\thisrow{c}] {\n"           +\
              "                  x y c\n"                                +\
              "                 " + x1s + space + y1s + space + z1s + "\n"  +\
              "                 " + x2s + space + y2s + space + z2s + "\n"  +\
              "                 " + x3s + space + y3s + space + z3s + "\n"  +\
              "    };\n\n"
    # shader=flat, or interp ...
    return string

#########################################################################################################
def rescaleXYZ(X, Y, Z):
    size = len(X)
    X = [ X[i] / (-10.0)  for i in range(size)]
    Y = [ Y[i] / 10.0     for i in range(size)]
    Z = [ Z[i] * 1000.0    for i in range(size)]
    return X, Y, Z


#########################################################################################################
def getTeX_precontent(X, Y, Z, labels, bottomSimplices, A, B, C):
    # iteration over every simplex, with data saved in iList, jList, and kList:
    s1 = ""; s2 = ""; s3 = ""; s4 = ""; s5 = ""

    lX = scr.getLoopedData(X, bottomSimplices)
    lY = scr.getLoopedData(Y, bottomSimplices)
    lZ = scr.getLoopedData(Z, bottomSimplices)

    for i in range( len(lX) ):
        x1, x2, x3, x4 = lX[i] # x4 == x1, it is looped around the simplex!!
        y1, y2, y3, y4 = lY[i] # y4 == y1, it is looped around the simplex!!
        z1, z2, z3, z4 = lZ[i] # z4 == z1, it is looped around the simplex!!

        s1 += getSimplexTeX(x1,y1,z1, x2,y2,z2, x3,y3,z3) # interpolates inside a simplex
        s2 += getLineTeX(x1, y1, x2, y2) + getLineTeX(x2, y2, x3, y3) + getLineTeX(x3, y3, x1, y1)
        s5 += getLineForFile(x1, y1, z1) + getLineForFile(x2, y2, z2) + getLineForFile(x3, y3, z3)
    ###

    lX_Ordered = scr.getOrderedData(X, bottomSimplices)
    lY_Ordered = scr.getOrderedData(Y, bottomSimplices)
    labels_Ordered = scr.getOrderedData(labels, bottomSimplices)

    for i in range( len(lX_Ordered) ):
        s3 += getCircleTeX(lX_Ordered[i], lY_Ordered[i])
        s4 += getLabelTeX(A, B, C, labels_Ordered[i], lX_Ordered[i], lY_Ordered[i])
    ###

    return (s1, s2, s3, s4, s5)

#########################################################################################################
def createFileStables(threeColumnValues, stablesFile):
    file = open(stablesFile, "w")
    file.write("X Y Z\n" + threeColumnValues)
    file.close()

#########################################################################################################
def createTeXfile(teXcontent, texName):
    file = open(texName, "w")
    file.write(teXcontent)
    file.close()

# #########################################################################################################
# def getColormap(colormap):
# 	if (colormap == "aflow"):
# 		return "colormap name = \\colormap"
# 	return "colormap/bluered"

#########################################################################################################
def getTeXcontent(s1, s2, s3, s4, stablesFile):
    teXcontent = """
    \\documentclass{standalone}\n
    \\usepackage[scaled]{helvet}
    \\renewcommand\\familydefault{\\sfdefault}\n
    \\usepackage[T1]{fontenc}
    \\usepackage{units}
    \\usepackage{amsmath}
    \\usepackage{chemformula}
    \\usepackage{tikz}
    \\usetikzlibrary{calc,fadings,decorations.pathreplacing}
    \\usepackage{units}
    \\usepackage{pgfplots}
    \\pgfplotsset{width=7cm,compat=newest}
    %%%<
    \\usepackage{verbatim}
    \\usepackage[active,tightpage]{preview}
    \\PreviewEnvironment{tikzpicture}
    \\setlength\\PreviewBorder{0pt}%
    %%%>
    \\pgfplotsset{
        % define the custom colormap
        colormap={aflow}{
            rgb255=(10, 62, 198),
            rgb255=(253, 152, 4),
        },
    }\n
    %%%%%%%%%%%%%%%%%
    %definitions
    \\newcommand\\scale{3.5}
    \\newcommand\\ymin{-1.3}
    \\newcommand\\ymax{9.5}
    \\newcommand\\xmin{-10.8}
    \\newcommand\\xmax{0.7}
    \\newcommand\\zmin{-475}
    \\newcommand\\zmax{0}
    \\newcommand\\colormap{aflow}
    %%%%%%%%%%%%%%%%%\n
    \\begin{document}\n
    \\begin{tikzpicture}
    \\begin{scope}[]
    \\begin{axis}[   scale=\\scale,
        %colormap/hot,
        % use defined custom colormap
        colormap name = \\colormap,
        colorbar,
        point meta min  = \\zmin,
        point meta max = \\zmax,
        colorbar style = {
        %				title = Formation enthalpy (meV/atom),
                        ylabel = Formation enthalpy (meV/atom),
                        ylabel style = {font=\\small},
                        at={ (0.87, 1) },
                        height = 0.55 * \\pgfkeysvalueof{/pgfplots/parent axis width}
                        },
        colorbar/width=1.8mm,
        unit vector ratio*=1 1 1,
        hide axis,
        ymin  = \\ymin,
        ymax = \\ymax,
        xmin  = \\xmin,
        xmax = \\xmax,
        ]\n\n
    """
    teXcontent += s1
    teXcontent += """    \\addplot[
            scatter,
           %  colormap/hot,
                % use defined custom colormap
                 colormap name = \\colormap,
                 only marks,
            point meta=\\thisrow{Z},
            point meta min= -0.5, %-0.5
            point meta max= 0.0, %0.2
        ]
        table[x=X,y=Y]{""" + stablesFile   + """}; \n\n\n"""
    teXcontent += s2 + "\n\n" + s3 + "\n\n\\end{axis} \n \\end{scope}\n\n"
    teXcontent +=  """ \\begin{scope}[ scale=\\scale]
          \\begin{axis}[
          unit vector ratio*=1 1 1,
          hide axis,\n ymin  = \\ymin,\n ymax = \\ymax,\n xmin  = \\xmin,\n xmax = \\xmax,\n ]\n\n"""
    teXcontent += s4 + "\n\end{axis}\n\end{scope}\n\end{tikzpicture}\n\end{document}\n"
    return teXcontent

#########################################################################################################
def getPDFconvexHull(X, Y, Z, labels, bottomSimplices, A, B, C, pdfName):
    import os
    # rescaling:
    X, Y, Z = rescaleXYZ(X, Y, Z)

    s1, s2, s3, s4, s5 = getTeX_precontent(X, Y, Z, labels, bottomSimplices, A, B, C)

    stablesFile = "stables_" + pdfName + ".txt"
    teXcontent = getTeXcontent(s1, s2, s3, s4, stablesFile)
    createFileStables(s5, stablesFile)

    texName = pdfName + ".tex"
    auxName = pdfName + ".aux"
    logName = pdfName + ".log"
    createTeXfile( teXcontent, texName )
    os.system("pdflatex " + texName)
    os.system("rm " + auxName)
#     os.system("rm " + logName)
#########################################################################################################



#########################################################################################################
def getPDFconvexHullPlusPoints(Xpoints, Ypoints, Zpoints, labPoints, X, Y, Z, labels, bottomSimplices, A, B, C, pdfName, scale):
    import os
    # rescaling:
    X, Y, Z = rescaleXYZ(X, Y, Z)
    Xpoints, Ypoints, Zpoints = rescaleXYZ(Xpoints, Ypoints, Zpoints)

    s1, s2, s3, s4, s5 = getTeX_precontent(X, Y, Z, labels, bottomSimplices, A, B, C)

    for i in range(len(Xpoints)):
        s3 += getCircleTeX(Xpoints[i], Ypoints[i])
        s5 += getLineForFile(Xpoints[i], Ypoints[i], Zpoints[i])
        # s4 += getLabelTeX(A, B, C, labPoints[i], Xpoints[i], Ypoints[i])
        s4 += getLabelTeX2(A, B, C, labPoints[i], Xpoints[i], Ypoints[i], scale)
    #

    stablesFile = "stables_" + pdfName + ".txt"
    teXcontent = getTeXcontent(s1, s2, s3, s4, stablesFile)
    createFileStables(s5, stablesFile)

    texName = pdfName + ".tex"
    auxName = pdfName + ".aux"
    logName = pdfName + ".log"
    createTeXfile( teXcontent, texName )
    os.system("pdflatex " + texName)
    os.system("rm " + auxName)
#     os.system("rm " + logName)
#########################################################################################################
