# import plotly.plotly as py
from chart_studio import plotly
import plotly.graph_objs as go
import numpy as np
# import TernaryHull.CalcScripts as scr
import CalcScripts as scr

######################################
def getTraceScatterPlotly(X,Y,Z,limits):
    return go.Scatter3d(
        showlegend=False,
        x = X, y = Y, z = Z,
        mode = 'markers',
        marker = dict(
                    cmin = limits[6], # = Emin
                    cmax = limits[7], # = Emax
                    size=1.5,
                    color=Z,
                    opacity=0.9,
                    colorscale='Jet',
            #                     colorScale='Jet',

                )
        )


######################################
#colorBarTitle='Formation Entalphy\n(eV/atom)'
def getTraceMeshPlotly(Xall, Yall, Zall, simplices, limits, colorBarTitle):
    XhullOrdered = scr.getOrderedData(Xall, simplices)
    YhullOrdered = scr.getOrderedData(Yall, simplices)
    ZhullOrdered = scr.getOrderedData(Zall, simplices)
    return go.Mesh3d(
            x = XhullOrdered, y = YhullOrdered, z = ZhullOrdered,
            cmin = limits[6], # = Emin
            cmax = limits[7], # = Emax
            opacity = 0.4,
            colorbar = go.ColorBar(
                        title     = colorBarTitle,
                        titleside = "bottom",
                        ),
    #         colorscale = [['0', 'rgb(255, 0, 0)'], 
    #                       ['0.5', 'rgb(0, 255, 0)'], 
    #                       ['1', 'rgb(0, 0, 255)']],
            colorscale = 'Jet', #'Viridis',
            intensity  = ZhullOrdered,
#             i = iList,
#             j = jList,
#             k = kList,
            lighting   = dict( ambient=1.0, diffuse=1.0,
                               specular=0.0, roughness=1.0,
                               fresnel=0.0)
            #name = 'z',
            #showscale = True
            )


######################################
def getLayoutPlotly(limits, annotations):
    [xmin,xmax, ymin,ymax, zmin, zmax, temp6,temp7] = limits
    return go.Layout(
                scene = dict(
                            xaxis = dict(visible = False, range = [xmin, xmax],), #option:title = "Co"
                            yaxis = dict(visible = False, range = [ymin, ymax],),
                            zaxis = dict(visible = False, range = [zmin, zmax], showgrid= False,\
                                         title = "Entalphy formation (meV)"),                    
                    
                            annotations = annotations,
                            camera = dict(
                                            up     = dict(x=1, y=0, z=0),
                                            center = dict(x=0, y=0, z=0),
                                            eye    = dict(x=0.0, y=0.0, z=-2),
                                        ),
                        ),
                margin=dict(
                            l=0,
                            r=0,
                           # b=0,
                            t=0
                            ),
                
        )
    
######################################
def getLinesPolygon(xSequence, ySequence, zSequence, limits):
    return go.Scatter3d(
                        x = xSequence,
                        y = ySequence,
                        z = zSequence,
                        mode   = 'lines+markers', #lines',
                        line   = dict(color='#000000', width=3),
                        marker = dict(
                                    cmin = limits[6], # = Emin
                                    cmax = limits[7], # = Emax
                                    size  = 5,
                                    color = zSequence,                                
#                                       color=[1,1,1],    # set color to an array/list of desired values
#                                         colorscale='Jet',   # choose a colorscale
                                    opacity=1,
                                    # I do not know why, but colors are correct related if width=2
                                    # line=dict(width=2, color='rgb(204, 204, 204)' ), 
                                    ),
                        showlegend=False
                        ) 

# ######################################  
# # simplices = hull.simplices
# def getOrderedIndicesOfConvexHull(simplices):
#     orderedIndices = []
#     for s in simplices:
#         orderedIndices.append( s[0] ) #index of firstVertex of simplex `s`
#         orderedIndices.append( s[1] ) #index of secondVertex
#         orderedIndices.append( s[2] ) #index of thirdVertex
    
#     # "set" will do UNION (avoiding repetition),
#     # and "list" will convert the result {} into a list [].
#     orderedIndices = sorted( list( set( orderedIndices ) ) )
#     return orderedIndices

# ######################################
# def getLoopedSimplicesInConvexHull(simplices):
#     loopedSimplices = []
#     for s in simplices:
#         loopedSimplices.append( [ s[0], s[1], s[2], s[0] ] ) # Loop over the indices of the simplex `s`
#     return loopedSimplices

# ######################################
# def getOrderedData(Xall, simplices):
#     orderedIndices = scr.getOrderedIndicesOfConvexHull(simplices)
#     return [ Xall[i] for i in orderedIndices ]

# ######################################
# def getLoopedData(Xall, simplices):
#     loopedSimplices = scr.getLoopedSimplicesInConvexHull(simplices)
#     loopedX = []
#     for i in range( len(loopedSimplices) ):  
#         x0 = Xall[ loopedSimplices[i][0] ]
#         x1 = Xall[ loopedSimplices[i][1] ]
#         x2 = Xall[ loopedSimplices[i][2] ]
#         x3 = Xall[ loopedSimplices[i][3] ]
#         loopedX.append( [ x0, x1, x2, x3 ] )
#     #
#     return loopedX

######################################
def getAnnotationsForPlotly(Xall, Yall, Zall, labelsAll, simplices):
    Xordered = scr.getOrderedData(Xall, simplices)
    Yordered = scr.getOrderedData(Yall, simplices)
    Zordered = scr.getOrderedData(Zall, simplices)
    labelsordered = scr.getOrderedData(labelsAll, simplices)
    
    annotations = []
    for i in range( len(Xordered) ):
        annotations.append(
                            dict(
                                   showarrow = False,
                                   x = Xordered[i],
                                   y = Yordered[i],
                                   z = Zordered[i],
                                   text = labelsordered[i],
                                   xanchor = "left",
                                   xshift = 8,
                                   opacity = 1.0,
                                  ),
                            )
    #
    return annotations  

######################################
def getLinesSimplicesPlotly(X, Y, Z, simplices, limits):
    lX = scr.getLoopedData(X, simplices)
    lY = scr.getLoopedData(Y, simplices)
    lZ = scr.getLoopedData(Z, simplices)
    return [ getLinesPolygon(lX[i], lY[i], lZ[i], limits) for i in range(len(simplices)) ]

######################################
def getTriangularBox(limits):
    import math
    Emin = limits[6]
    Emax = limits[7]
    ym   = 100.0 * math.sqrt(3.0) / 2.0
    return go.Scatter3d(
            showlegend=False,
            x = [0, 100, 50,  0,   0,100,50,0,  0, 100, 50, 0,   100, 100, 50, 50   ],
            y = [0, 0,   ym,  0,   0,  0,ym,0,  0, 0,   ym, 0,     0,   0, ym, ym   ],
            z = [Emax,Emax,Emax,Emax, 0,0,0,  0,  Emin,Emin,Emin,Emin, Emin,Emax,Emax,Emin  ],
            mode='lines',
            line=dict(color='#000000'),    
            marker=dict(
                size=8,
#               color=[1,1,1],                # set color to an array/list of desired values
#               colorscale='Jet',   # choose a colorscale
                opacity=1,
                # I do not know why, but colors are correct related if width=2
                #         line=dict(width=2, color='rgb(204, 204, 204)' ), 
                )
                            
            )

######################################
def make3DPlotlyDouble(Xall,Yall,Zall,labelsAll,bottomSimplices,\
                       limits, colorBarTitle,\
                       nameHTMLfile): #boxCorners = [xmin,xmax, ymin,ymax, zmin, zmax]
    from chart_studio import plotly
    import plotly.graph_objs as go
    import numpy as np

    trace1 = getTraceScatterPlotly(Xall, Yall, Zall, limits)
    trace2 = getTraceMeshPlotly(Xall,Yall,Zall, bottomSimplices, limits,colorBarTitle)
    trace3 = getTriangularBox(limits)
    lines  = getLinesSimplicesPlotly(Xall, Yall, Zall, bottomSimplices, limits)    
    annotations= getAnnotationsForPlotly(Xall, Yall, Zall, labelsAll, bottomSimplices)    

    data   = [trace1, trace2, trace3] + lines  #[trace4, trace5, trace6] #trace2
    layout = getLayoutPlotly(limits, annotations)

    #------------------------------------------------------------
    # plotting:
    fig = go.Figure(data=data, layout=layout)
    
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    plot(fig, filename=nameHTMLfile)
######################################

def make3DPlotly(Xtest, Ytest, Ztest, Xall,Yall,Zall, labelsAll,simplices,\
                       limits, colorBarTitle,\
                       nameHTMLfile): #boxCorners = [xmin,xmax, ymin,ymax, zmin, zmax]
    from chart_studio import plotly
    import plotly.graph_objs as go
    import numpy as np

    trace0 = getTraceScatterPlotly(Xtest, Ytest, Ztest, limits)
#     trace1 = getTraceScatterPlotly(Xall, Yall, Zall, limits)
#     trace2 = getTraceMeshPlotly(Xall,Yall,Zall, simplices, limits,colorBarTitle)
    lines  = getLinesSimplicesPlotly(Xall, Yall, Zall, simplices, limits)    
    annotations= getAnnotationsForPlotly(Xall, Yall, Zall, labelsAll, simplices)    

    data   = [trace0] + lines  #[trace4, trace5, trace6] #trace2
    layout = getLayoutPlotly(limits, annotations)

    #------------------------------------------------------------
    # plotting:
    fig = go.Figure(data=data, layout=layout)
    
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    plot(fig, filename=nameHTMLfile)

######################################
def makeAsimple3DPlotly(X, Y, Z,\
                       limits, colorBarTitle,\
                       nameHTMLfile): #boxCorners = [xmin,xmax, ymin,ymax, zmin, zmax]
    from chart_studio import plotly
    import plotly.graph_objs as go
    import numpy as np

    trace0 = getTraceScatterPlotly(X, Y, Z, limits)
#     trace1 = getTraceScatterPlotly(Xall, Yall, Zall, limits)
#     trace2 = getTraceMeshPlotly(Xall,Yall,Zall, simplices, limits,colorBarTitle)
    trace3 = getTriangularBox(limits)
    # lines  = getLinesSimplicesPlotly(Xall, Yall, Zall, simplices, limits)    
    # annotations= getAnnotationsForPlotly(Xall, Yall, Zall, labelsAll, simplices)    

    data   = [trace0, trace3] #+ lines  #[trace4, trace5, trace6] #trace2
    layout = getLayoutPlotly(limits, [])  # limits, annotations

    #------------------------------------------------------------
    # plotting:
    fig = go.Figure(data=data, layout=layout)
    
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    plot(fig, filename=nameHTMLfile)

######################################
def makeAsimple3DPlotly(X, Y, Z,\
                       limits, colorBarTitle,\
                       nameHTMLfile): #boxCorners = [xmin,xmax, ymin,ymax, zmin, zmax]
    from chart_studio import plotly
    import plotly.graph_objs as go
    import numpy as np

    zeros = [0.0 for i in range(len(X))]
    
    trace0 = go.Scatter3d(
        showlegend=False,
        x = X, y = Y, z = Z,
        mode = 'markers',
        marker = dict(
                    cmin = limits[6], # = Emin
                    cmax = limits[7], # = Emax
                    size=2.0,
                    color=Z,
                    opacity=0.9,
                    colorscale='Jet',
                    colorbar = dict(title = colorBarTitle,
                                    titleside = "bottom")            

            #                     colorScale='Jet',

                )        
        )
    
    trace0 = getTraceScatterPlotly(X, Y, Z, limits)
#     trace1 = getTraceScatterPlotly(Xall, Yall, Zall, limits)
    trace2 = go.Mesh3d(
            x = X, y = Y, z = zeros,
            cmin = limits[6], # = Emin
            cmax = limits[7], # = Emax
            opacity = 1.0,
            colorbar = go.ColorBar(
                        title     = colorBarTitle,
                        titleside = "bottom",
                        ),
    #         colorscale = [['0', 'rgb(255, 0, 0)'], 
    #                       ['0.5', 'rgb(0, 255, 0)'], 
    #                       ['1', 'rgb(0, 0, 255)']],
            colorscale = 'Jet', #'Viridis',
            intensity  = Z,
#             i = iList,
#             j = jList,
#             k = kList,
            lighting   = dict( ambient=1.0, diffuse=1.0,
                               specular=0.0, roughness=1.0,
                               fresnel=0.0)
            #name = 'z',
            #showscale = True
            )
    trace3 = getTriangularBox(limits)
    # lines  = getLinesSimplicesPlotly(Xall, Yall, Zall, simplices, limits)    
    # annotations= getAnnotationsForPlotly(Xall, Yall, Zall, labelsAll, simplices)    

    data   = [trace2, trace3] #+ lines  #[trace4, trace5, trace6] #trace2
    layout = getLayoutPlotly(limits, []) # limits, annotations

    #------------------------------------------------------------
    # plotting:
    fig = go.Figure(data=data, layout=layout)
    
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    plot(fig, filename=nameHTMLfile)

######################################
def make3DPlotlyDoublePlane(Xall,Yall,Zall,labelsAll,bottomSimplices,\
                       limits, colorBarTitle,\
                       nameHTMLfile): #boxCorners = [xmin,xmax, ymin,ymax, zmin, zmax]
    from chart_studio import plotly
    import plotly.graph_objs as go
    import numpy as np

    zeros = [0.0 for i in range(len(Xall))]

    trace1 = getTraceScatterPlotly(Xall, Yall, Zall, limits)
    # trace2 = getTraceMeshPlotly(Xall,Yall,Zall, bottomSimplices, limits,colorBarTitle)

#     trace2 = go.Mesh3d(
#             x = Xall, y = Yall, z = zeros,
#             cmin = limits[6], # = Emin
#             cmax = limits[7], # = Emax
#             opacity = 1.0,
#             colorbar = go.ColorBar(
#                         title     = colorBarTitle,
#                         titleside = "bottom",
#                         ),
#     #         colorscale = [['0', 'rgb(255, 0, 0)'], 
#     #                       ['0.5', 'rgb(0, 255, 0)'], 
#     #                       ['1', 'rgb(0, 0, 255)']],
#             colorscale = 'Jet', #'Viridis',
#             intensity  = Zall,
# #             i = iList,
# #             j = jList,
# #             k = kList,
#             lighting   = dict( ambient=1.0, diffuse=1.0,
#                                specular=0.0, roughness=1.0,
#                                fresnel=0.0)
#             #name = 'z',
#             #showscale = True
#             )



    trace3 = getTriangularBox(limits)
    lines  = getLinesSimplicesPlotly(Xall, Yall, Zall, bottomSimplices, limits)    
    annotations= getAnnotationsForPlotly(Xall, Yall, Zall, labelsAll, bottomSimplices)    

    data   = [trace1, trace3] + lines  #[trace4, trace5, trace6] #trace2
    layout = getLayoutPlotly(limits, annotations)

    #------------------------------------------------------------
    # plotting:
    fig = go.Figure(data=data, layout=layout)
    
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    plot(fig, filename=nameHTMLfile)


