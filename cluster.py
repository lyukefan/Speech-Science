from segment import *
from lcma import *
from config import *

import numpy as np
import scipy

# segments are a list of short mfcc feature - time sequences
def extract_node(segments):
    n_segments = len(segments)
    similarity_scores = [np.zeros((seg.shape[1],)) for seg in segments]
    for i in range(n_segments):
        for j in range(i+1, n_segments):
            paths = compare_signal(segments[i], segments[j])
            for path in paths.values:
                for coord in path:
                    similarity_scores[i][coord[0]] += 1
                    similarity_scores[j][coord[1]] += 1 # to be replace by a score

    # instead of triangular averaing, we use gaussian for simplicity
    smoothed_similarity = [
        scipy.ndimage.filters.gaussian_filter(score, sigma=10, mode='nearest') 
        for score in similarity_scores
    ]

    local_extremas = [
        scipy.signal.argrelextrema(sim, np.greater, order=1, mode='clip')
        for sim in smoothed_similarity
    ]

    n_node = reduce(lambda x, y: x+y, map(len, local_extremas))

    # build edges



def build_graph():
    pass

# E is represented by a set of 3-tuple (v1, v2, weight), 
# v1 and v2 are integers in range(0, n_node) 
# should return a set of connect-component as described
# in sect IV. B.

# the use of set: 
# you can access all the elements in an unordered set by:
# for e in E:
#     func(e)
# you may also convert E to another data-structure first 
# if you need to
def cluster(n_node, E):
    sum_weight=sum([e.weight for e in E ])
    for i in range(0,n_node)
      newman_id[i]=i
      a[i]=0
    for e in E
      newman_e[e.x][e.y]=e.weight/sum_weight
      newman_e[e.y][e.x]=e.weight/sum_weight
      newman_a[e.x]=newman_a[e.x]+neman_e[e.x][e.y]
      newman_a[e.y]=newman_a[e.y]+neman_e[e.x][e.y]
    modularity_Q=0
    for i in range(0,n_node)
        Q=Q+newman[i][i]-newman_a[i]*newman_a[i];
    for i in range(0,n_node)
        maxdelta=0
        u=-1
        v=-1
        for e in E
            if (newman_id[e.x]=newman_id[e.y]):
                continue
            elif 2*newman_e[newman_id[e.x]][newman_id[e.y]]-2*newman_a[newman_id[e.x]]*newman_a[newman_id[e.y]]>madelta:
                delta=2*newman_e[newman_id[e.x]][newman_id[e.y]]-2*newman_a[newman_id[e.x]]*newman_a[newman_id[e.y]]
                u=newman_id[e.x]
                v=newman_id[e.y]
#        if Q+maxdelta<bound  stopfor
        for j in range(0,n_node)
            if newman_id[i]=v:
                newman_id[i]=u
            elif newman_id[i]=i:
                newman_e[u][i]=newman_e[u][i]+newman_e[v][i]
                newman_e[i][u]=newman_e[u][i]
        newman_a[u]=newman_a[u]+newman_a[v];
                

        
        
            
            
        

