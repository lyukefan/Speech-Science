import numpy as np
import librosa as rosa
import matplotlib.pyplot as plt
from config import *

def extract_feature(wav_segment, sr):
    mfccs = rosa.feature.mfcc(y=wav_segment[0:sr], sr=sr, n_mfcc=30)
    # mfccs: (N, T)
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_vars = np.std(mfccs, axis=1)
    whitened_mfccs = (mfccs - mfcc_means[:, None]) / mfcc_vars[:, None]
    whitened_mfcc_norms = np.linalg.norm(whitened_mfccs, axis=0)
    normalized_mfccs = whitened_mfccs / whitened_mfcc_norms[None, :]
    return normalized_mfccs

def dump_feature(name, feature):
    pass

def compare_signal(x, y):
    # x and y are [n_mfcc, n_time_samples] arrays
    difference = x[:, :, None] - y[:, None, :]
    print(difference.shape)
    cost_table = np.linalg.norm(difference, axis=0)
    print(cost_table.shape)
    paths = segmental_DTW(cost_table, R=3)
    _visualize_paths(cost_table, paths)
    # plt.matshow(cost_table)
    # plt.show()
    
    # path refinement
    for key, path in paths.items():
        # using <s1, s2, ..., sn> notations as in the paper
        S = np.array([cost_table[coord[0], coord[1]] for coord in path])
        (refined_path_start, refined_path_end) = LCMA(S, L=25)
        paths[key] = paths[key][refined_path_start:refined_path_end]

    _visualize_paths(cost_table, paths)

# returns several paths for future refinement (done by LCMA)
# assuming R >= 1
def segmental_DTW(cost_table, R):
    DTW_table = np.zeros_like(cost_table)
    m, n = cost_table.shape
    region_identifiers = np.array([
        [(i-j+R)//(2*R+1) for j in range(n)] for i in range(m)
    ])
    # print(region_identifiers)
    identifiers = {(i-j+R)//(2*R+1) for j in range(n) for i in range(m)}
    start_points = { i:(max(0,(2*R+1)*i), max(0,-(2*R+1)*i)) for i in identifiers}
    paths = { i:[] for i in identifiers if start_points[i][0] < m and start_points[i][1] < n}
    # determine if (i, j) is possible to appear in a path
    reachable = np.array([
        [ i >= start_points[region_identifiers[i,j]][0] and j >= start_points[region_identifiers[i,j]][1]
            for j in range(n) ] for i in range(m)
    ])
    # for tracing back the paths
    last = np.zeros_like(cost_table)

    # print(reachable)
    
    DTW_table[0, 0] = cost_table[0, 0]

    PI, PJ, PIJ = 1, 2, 3
    for i in range(1, m):
        cell_region_indentifier = region_identifiers[i, 0]
        if not reachable[i, 0]:
            DTW_table[i, 0] = 0
        else:
            DTW_table[i, 0] = DTW_table[i-1, 0] + cost_table[i, 0]
            last[i, 0] = PI

    for j in range(1, n):
        cell_region_indentifier = region_identifiers[0, j]
        if not reachable[0, j]:
            DTW_table[0, j] =  0
        else:
            DTW_table[0, j] = DTW_table[0, j-1] + cost_table[0, j]
            last[0, j] = PJ
    # print(DTW_table)


    for i in range(1, m):
        for j in range(1, n):
            if not reachable[i, j]:
                DTW_table[i, j] = -1
                continue
            min_prev = INFINITY
            if reachable[i-1, j] and (region_identifiers[i-1,j] == region_identifiers[i,j]):
                if DTW_table[i-1, j] < min_prev:
                    min_prev = DTW_table[i-1, j]
                    last[i, j] = PI
            if reachable[i-1,j-1] and (region_identifiers[i-1,j-1] == region_identifiers[i,j]):
                if DTW_table[i-1,j-1] < min_prev:
                    min_prev = DTW_table[i-1, j-1]
                    last[i,j] = PIJ
            if reachable[i,j-1] and (region_identifiers[i,j-1] == region_identifiers[i,j]):
                if DTW_table[i,j-1] < min_prev:
                    min_prev = DTW_table[i,j-1]
                    last[i,j] = PJ
            DTW_table[i, j] = min_prev + cost_table[i, j]
    # print(last)
    # print(DTW_table)

    # reconstruct paths using last and DTW_table
    minimum_costs = { i:INFINITY for i in identifiers}
    destinations = { i:(0, 0) for i in identifiers}
    outer_cells = [(i, n-1) for i in range(m)] + [(m-1, j) for j in range(n-1)]
    for i, j in outer_cells:
        if DTW_table[i,j] < minimum_costs[region_identifiers[i,j]]:
            destinations[region_identifiers[i,j]] = (i, j)
            minimum_costs[region_identifiers[i,j]] = DTW_table[i,j]
    # print(region_identifiers)
    # print(destinations)

    for r in paths.keys():
        cursor_x, cursor_y = destinations[r]
        start_point = start_points[r]
        while (cursor_x, cursor_y) != start_point:
            paths[r] = [(cursor_x, cursor_y)] + paths[r]
            if last[cursor_x, cursor_y] == PI:
                cursor_x, cursor_y = cursor_x-1, cursor_y
            elif last[cursor_x, cursor_y] == PJ:
                cursor_x, cursor_y = cursor_x, cursor_y-1
            elif last[cursor_x, cursor_y] == PIJ:
                cursor_x, cursor_y = cursor_x-1, cursor_y-1
            else:
                print(r, cursor_x, cursor_y)
                print(paths[r])
                assert False
        paths[r] = [start_point] + paths[r]

    # for k, v in paths.items():
    #     print(k, v)
    
    return paths
            
def _print_path(m, n, path):
    mat = np.zeros((m,n))
    for i,j in path:
        mat[i,j] = 1
    print(mat)

# brute force search (O(NL)), instead of O(N log(L)) algorithm
# returns a pair of left-inclusive indices
def LCMA(S, L):
    n_samples = S.shape[0]
    minimum_pair = (-1, -1)
    if n_samples < L:
        return minimum_pair
    current_minimum = 1e10
    # l < 2L-1, see [27], lemma 7
    for l in range(L, L*2):
        for cursor in range(0, n_samples - l):
            mean = np.mean(S[cursor:cursor+l])
            if mean < current_minimum:
                current_minimum = mean
                minimum_pair = (cursor, cursor+l)
    return minimum_pair

def _compare_test():
    wav1, sr1 = rosa.core.load(SEGMENTED_PATH+'32.wav')
    wav2, sr2 = rosa.core.load(SEGMENTED_PATH+'33.wav')
    mfcc1, mfcc2 = extract_feature(wav1, sr1), extract_feature(wav2, sr2)
    plt.matshow(mfcc1)
    plt.matshow(mfcc2)
    plt.show()
    compare_signal(mfcc1, mfcc2)

def _visualize_paths(cost_table, paths):
    cost_table_copy = np.copy(cost_table)
    for path in paths.values():
        for coord in path:
            cost_table_copy[coord[0], coord[1]] = 0
    plt.matshow(cost_table_copy)
    plt.show()

if __name__ == '__main__':
    _compare_test()
