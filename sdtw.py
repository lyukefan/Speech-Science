import numpy as np
import librosa as rosa

def extract_feature(wav_segment):
    pass

def dump_feature(name, feature):
    pass

def whiten(features, means, vars):
    pass

# returns several paths for future refinement (done by LCMA)
def segmental_DTW(x, y):
    pass

# returns the path defined by [(x1, y1), (x2, y2), ...]
# as a subroutine of segmental_DTW
def DTW(x, y, start):
    pass

# brute force search (O(NL)), instead of O(N log(L)) algorithm
# returns a pair of left-inclusive indices
def LCMA(S, L):
    n_samples = S.shape[0]
    minimum_pair = (-1, -1)
    current_minimum = 1e10
    # l < 2L-1, see [27], lemma 7
    for l in range(L, L*2):
        for cursor in range(0, n_samples - l):
            mean = np.mean(S[cursor:cursor+l])
            if mean < current_minimum:
                current_minimum = mean
                minimum_pair = (cursor, cursor+l)
    return minimum_pair

if __name__ == '__main__':
    s = np.array([1,2,3,1,0,1,1,0,3])
    p = LCMA(s, 3)
    print(s[p[0]:p[1]])
