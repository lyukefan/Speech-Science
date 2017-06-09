import librosa as rosa
import numpy as np

# WAV_FILE_PATH = 'data/audio.wav'
WAV_FILE_PATH = 'data/short.wav'
OUTPATH = 'data/segmented/'

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
    print(sr)
    print(wav.shape)
    intervals = rosa.effects.split(wav, top_db=50)
    n_intervals = intervals.shape[0]
    print(n_intervals, intervals.shape)
    for i in range(n_intervals):
        start, end = intervals[i]
        print(start, end)
        write_wav('%s%d.wav'%(OUTPATH, i), wav[intervals[i][0]:intervals[i][1]], sr)

if __name__ == '__main__':
    segment(WAV_FILE_PATH)
    # gen_sample(WAV_FILE_PATH)

