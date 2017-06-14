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
    pass

