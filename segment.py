import librosa as rosa
import numpy as np

# WAV_FILE_PATH = 'data/audio.wav'
WAV_FILE_PATH = 'data/short.wav'
OUTPATH = 'data/segmented/'
SILENCE_THRESHOLD = 42 # dB
MINIMUN_SEGMENT_DURATION = 0.5 # sec

def write_wav(out_path, S, sr):
    maxv = np.iinfo(np.int16).max
    rosa.output.write_wav(out_path, (S * maxv).astype(np.int16), sr)

''' generate a short (5min) sample of audio file for use of debugging '''
def gen_sample(wav_file):
    wav, sr = rosa.core.load(wav_file, mono=True)
    maxv = np.iinfo(np.int16).max
    rosa.output.write_wav('data/short2.wav', (wav * maxv).astype(np.int16)[300*sr:600*sr], sr)


''' devide the audio file into non-silent segments'''
def segment(wav_file):
    wav, sr = rosa.core.load(wav_file, mono=True)
    intervals = rosa.effects.split(wav, top_db=SILENCE_THRESHOLD)
    n_samples = wav.shape[0]
    n_intervals = intervals.shape[0]
    max_len, min_len = 0, 36000 * sr
    for i in range(n_intervals):
        start, end = intervals[i]
        duration = end - start
        max_len = max(duration, max_len)
        min_len = min(duration, min_len)
        if duration/sr > MINIMUN_SEGMENT_DURATION:
            write_wav('%s%d.wav'%(OUTPATH, i), wav[intervals[i][0]:intervals[i][1]], sr)
        else:
            write_wav('%s_%d.wav'%(OUTPATH, i), wav[intervals[i][0]:intervals[i][1]], sr)
    
    print('divided into %d segments, max_length=%fs, min_length=%fs, avg=%fs'
             % (n_intervals, max_len/sr, min_len/sr, n_samples/sr/n_intervals))

if __name__ == '__main__':
    segment(WAV_FILE_PATH)

