from matplotlib import pyplot as plt
from matplotlib import patches as ptch
from matplotlib import lines as ln
from matplotlib import rcParams as param
import numpy as np
# from matplotlib.patches import Wedge
# # from plotly.graph_objs import *
# from matplotlib.gridspec import GridSpec


#const
radius = 1 #radius for circles at intersections
p_radius = 2 #radius for the pieces
origin = (0,0)
step = (5,5)
width = 50
wstep = 10

param.update({'font.size': 7})
#board plot


def board(radius, origin, step, width, wstep):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    #draw the square

    rectangle = plt.Rectangle(origin,width + wstep,width + wstep, fc =  None, ec = 'black', fill = False)
    ax.add_patch(rectangle)
    rectangle = plt.Rectangle(tuple(np.add(np.array(origin), np.array((1,1))).tolist()),width + wstep -2,width + wstep - 2, fc =  None, ec = 'black', fill = False)
    ax.add_patch(rectangle)



    #draw the lines 
    corners = []

    #3 rectangles and add their corners to possible pieces locations
    origin = tuple(np.add(np.array(origin), np.array(step)).tolist())
    origin = tuple(np.add(np.array(origin), np.array(step)).tolist())
    width = width-wstep
    rect_big = plt.Rectangle(origin,width, width, fc = None, ec = 'black', fill = False)
    corners.append(origin)

    corner = tuple(np.add(np.array(origin), np.array((width,0))).tolist())
    corners.append(corner)

    corner = tuple(np.add(np.array(origin), np.array((width,width))).tolist())
    corners.append(corner)
    corner = tuple(np.add(np.array(origin), np.array((0,width))).tolist())
    corners.append(corner)
    beg_line = [[origin[0], width//2 + origin[1]], [origin[0] + width, width//2 +  origin[1]], [width//2 +  origin[1], origin[0]], [width//2 +  origin[1], origin[0] +width]]



    origin = tuple(np.add(np.array(origin), np.array(step)).tolist())
    width = width-wstep
    corners.append(origin)
    corner = tuple(np.add(np.array(origin), np.array((width,0))).tolist())
    corners.append(corner)
    corner = tuple(np.add(np.array(origin), np.array((width,width))).tolist())
    corners.append(corner)
    corner = tuple(np.add(np.array(origin), np.array((0,width))).tolist())
    corners.append(corner)
    rect_mid = ptch.Rectangle(origin,width, width, fc = None, ec = 'black', fill = False)


    origin = tuple(np.add(np.array(origin), np.array(step)).tolist())
    width = width-wstep
    corners.append(origin)
    corner = tuple(np.add(np.array(origin), np.array((width,0))).tolist())
    corners.append(corner)
    corner = tuple(np.add(np.array(origin), np.array((width,width))).tolist())
    corners.append(corner)
    corner = tuple(np.add(np.array(origin), np.array((0,width))).tolist())
    corners.append(corner)
    rect_small = ptch.Rectangle(origin,width, width, fc =  None, ec = 'black', fill = False)
    end_line = [[origin[0], width//2 + origin[1]], [origin[0] + width, width//2 +  origin[1]], [width//2 +  origin[1], origin[0]], [width//2 +  origin[1], origin[0] +width]]    
            
    ax.add_patch(rect_big)
    ax.add_patch(rect_mid)
    ax.add_patch(rect_small)


    #4 lines
    for i in range(4):
        #line = ln.Line2D([0,0], [10,10], linewidth = 2)
        x = [beg_line[i][0], end_line[i][0]]
        y = [beg_line[i][1], end_line[i][1]]
        line = ln.Line2D(x, y, linewidth = 1, color = 'black', zorder = 1)
        ax.add_artist(line)
        corners.append((beg_line[i][0],beg_line[i][1] ))    
        if i >=2 :
            if i == 3:
                corners.append((beg_line[i][0],beg_line[i][1]- step[1] ))
            else : 
                corners.append((beg_line[i][0],beg_line[i][1]+ step[1] ))
        else :
            if i == 1:
                corners.append((beg_line[i][0]- step[0],beg_line[i][1]) )
            else :
                corners.append((beg_line[i][0]+ step[0],beg_line[i][1]) )
        corners.append((end_line[i][0],end_line[i][1]))

    #order corners by ascending x
    corners.sort(key = lambda elt: elt[0])
    corners.sort(key = lambda elt: elt[1], reverse=True)        
    #draw a grey circle at each intersection
    i = 0
    for corn in corners :
        circle = plt.Circle(corn, radius, fc = 'grey')
        ax.add_patch(circle)
        plt.text(corn[0], corn[1], str(i), )
        i+=1

    return ax, corners

def add_pieces(ax, corners, w_pieces, b_pieces):
    #add pieces 

    for elt in w_pieces :
        circle = plt.Circle(corners[elt], p_radius, fc = 'white', ec = 'black')
        ax.add_patch(circle)
        
    for elt in b_pieces :
        circle = plt.Circle(corners[elt], p_radius, fc = 'black', ec = 'white')
        ax.add_patch(circle)
        

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.axis('scaled')    
    plt.show()    


def save_state(filename, w_pieces, b_pieces, step):
    try :
        file = open(filename, mode = 'x')
    except FileExistsError :
        file = open(filename, mode = 'a' )
    if len(w_pieces) == 0 :
        if  len(b_pieces) == 0 : 
            line = str(step) + ';' + 'empty' + ';' + 'empty' + '\n'
        else :   
            line = str(step) + ';' + 'empty' + ';' + str(b_pieces) + '\n'
    if len(b_pieces) == 0 :
         line = str(step) + ';' + str(w_pieces) + ';'+ 'empty' + '\n'
    else :
        line = str(step) + ';' + str(w_pieces) + ';' + str(b_pieces) + '\n'
        
    file.write(line)
    file.close()
    
def extract_state(filename, step):
    file = open(filename, 'r')
    lines = file.read()
    lines.split()
    lines = lines.splitlines()
    step, w_pieces, b_pieces = lines[step].split(';')
    if w_pieces == 'empty':
        w_pieces = []
    if b_pieces == 'empty':
        b_pieces = []
    return w_pieces, b_pieces
    
# w_pieces = [1,7,15]
# b_pieces = [3,4,14]
# axes, corner = board(radius, origin, step, width, wstep)
# add_pieces(axes, corner, w_pieces, b_pieces)
# save_state('games/game01.txt', w_pieces, b_pieces, 0)
w_pieces, b_pieces = extract_state('games/game01.txt',0)
print(w_pieces, b_pieces)
# axes, corner = board(radius, origin, step, width, wstep)
# add_pieces(axes, corner, w_pieces, b_pieces)
plt.close('all')