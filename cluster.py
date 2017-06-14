from segment import *
from sdtw import *
from config import *
from functools import reduce
from itertools import accumulate

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# segments are a list of short mfcc feature - time sequences
def extract_node(segments):
    n_segments = len(segments)
    segment_lengths = list(map(lambda s: s.shape[1], segments))
    accumulate_lengths = [0] + list(accumulate(segment_lengths))
    total_length = accumulate_lengths[-1]

    print(total_length, accumulate_lengths)
    
    # sparse matrix representation of the graph
    row, col, data = [], [], []


    similarity_scores = [np.zeros((seg.shape[1],)) for seg in segments]
    for i in range(n_segments):
        for j in range(i+1, n_segments):
            paths = compare_signal(segments[i], segments[j])
            # print(paths)
            for path, average_distortion in paths.values():
                for coord in path:
                    row += [convert_to_global_index(i, coord[0], accumulate_lengths)]
                    col += [convert_to_global_index(j, coord[1], accumulate_lengths)]
                    data += [similarity_score(average_distortion)]

    similarity_coo = sp.coo_matrix((data, (row, col)), shape=(total_length, total_length))
    sim = similarity_coo.toarray()
    sim += np.transpose(sim)
    plt.matshow(sim[0:500, 0:500])
    plt.show()

    return similarity_coo, accumulate_lengths

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

# get a global index for j-th feature of fragment i
def convert_to_global_index(i, j, segments_acc):
    return segments_acc[i] + j

# implements eq. 9
def similarity_score(average_distortion):
    return (THETA - average_distortion) / THETA

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

if __name__ == '__main__':
    extract_node(load_feature())
