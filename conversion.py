import os
import glob
import time
import json

import numpy as np
from tqdm import tqdm

import librosa
import librosa.display
from librosa.feature import melspectrogram

from hparams import hparams

assert hparams.name == "wavenet_vocoder"

print('Hop size: {}'.format(hparams.hop_size))

################################################################################

def _normalize(S):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)

def pad_len(chunk, chunk_len):
    len_pad = 0
    if (chunk.shape[0] != chunk_len):
        len_pad = chunk_len - chunk.shape[0]
        chunk = librosa.util.fix_length(chunk, chunk_len)
    return chunk, len_pad

def store_data(out_dir, raw_samples, target_samples, Y, X, idx):
    raw_samples = np.array(raw_samples)
    target_samples = np.array(target_samples)
    Y = np.array(Y)
    X = np.array(X)

    np.save(os.path.join(out_dir,'raw_samples_{:05d}.npy'.format(idx)), raw_samples)
    np.save(os.path.join(out_dir,'target_samples_{:05d}.npy'.format(idx)), target_samples)
    np.save(os.path.join(out_dir,'target_mel_{:05d}.npy'.format(idx)), Y)
    np.save(os.path.join(out_dir,'raw_mel_{:05d}.npy'.format(idx)), X)

def convert_and_store_data(data_path='./dataset/LJSpeech-1.1/wavs/', out_dir="./dataset/ljspeech", sr=8000, sr_target=22050, chunk_len = 20480, sample_per_files=131):
    print('Check if data exist...')
    if os.path.isfile(os.path.join(out_dir,'data.json')):
        print('Numpy chuncks yet exists.\nFinished!')
        return
    else:
        print('Data not found. Start conversion...')

    files = glob.glob(data_path+"*.wav")
    raw_samples = []
    target_samples = []
    Y = []
    X = []
    count_sample = 0
    n_sample = 0
    data_idx = 0

    for f in tqdm(files):
        sample_chunks = []
        sample_chunks_hat = []
        sample_chunks_mel = []
        sample_chunks_mel_hat = []

        sample, sr = librosa.core.load(f, sr=sr_target, mono=True)
        length = sample.shape[0]

        # Adjust length of sample, multiple of hop_size
        len_padds = 0
        for i in range(0, sample.shape[0], chunk_len):
            chunk = sample[i:i+chunk_len]
            chunk, len_pad = pad_len(chunk, chunk_len)
            len_padds += len_pad

            chunk_mel = _normalize(melspectrogram(chunk, sr=sr_target, n_fft=hparams.fft_size, n_mels=hparams.num_mels, hop_length=hparams.hop_size)).astype(np.float32)
            chunk_hat = librosa.core.resample(chunk, sr_target, sr)
            chunk_mel_hat = _normalize(melspectrogram(chunk_hat, sr=sr, n_fft=hparams.fft_size, n_mels=hparams.num_mels, hop_length=hparams.hop_size)).astype(np.float32)

            sample_chunks.append(chunk)
            sample_chunks_hat.append(chunk_hat)
            sample_chunks_mel.append(chunk_mel)
            sample_chunks_mel_hat.append(chunk_mel_hat)

            N = chunk_mel.shape[0]
            assert len(chunk) >= N * hparams.hop_size

            N_hat = chunk_mel_hat.shape[0]
            assert len(chunk_hat) >= N_hat * hparams.hop_size

        assert len(sample_chunks) == (sample.shape[0]+len_padds)/chunk_len

        raw_samples.append(sample_chunks)
        target_samples.append(sample_chunks_hat)
        Y.append(sample_chunks_mel)
        X.append(sample_chunks_mel_hat)
        n_sample += 1

        # Store to file
        if n_sample % sample_per_files == 0:
            store_data(out_dir, raw_samples, target_samples, Y, X, data_idx)
            # print('Stored {:05d} partial.'.format(data_idx))
            raw_samples = []
            target_samples = []
            Y = []
            X = []
            data_idx += 1

    # Store last chucnks of samples
    if n_sample % sample_per_files > 0:
        store_data(out_dir, raw_samples, target_samples, Y, X, data_idx)

    data = {
        "n_sample" : n_sample,
        "sample_rate" : sr,
        "sample_rate_target" : sr_target,
        "chunk_len" : chunk_len,
        "sample_per_files" : sample_per_files,
        "timestamp" : time.time()
    }

    with open(os.path.join(out_dir,'data.json'), 'w') as f:
        json.dump(data, f)
    with open(os.path.join(out_dir,'sample_index.json'), 'w') as f:
        json.dump(data, f)
    print('Finished!')


if True:
    convert_and_store_data(data_path='./dataset/LJSpeech-1.1/wavs/', out_dir="./dataset/ljspeech")
