import os
import json
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import librosa
from librosa.feature import melspectrogram

from hparams import hparams

class LJSpeechDataset(Dataset):
    """`LJSpeech <https://keithito.com/LJ-Speech-Dataset/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        transform (callable, optional): A function/transform that  takes in an raw audio
            and returns a transformed version. E.g, ``transforms.Scale``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.data_mels = []
        self.target_mels = []
        self.data_samples = []
        self.target_samples = []

        self.chunk_size = 131
        self.num_samples = 0
        self.cached_pt = 0

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        self._read_info()

        #load first chunk of data
        self.data_mels = np.load(os.path.join(
          self.root, "raw_mel_{:05d}.npy".format(self.cached_pt)), allow_pickle=True)
        self.target_mels = np.load(os.path.join(
          self.root, "target_mel_{:05d}.npy".format(self.cached_pt)), allow_pickle=True)
        #load first chunk of data
        self.data_samples = np.load(os.path.join(
          self.root, "raw_samples_{:05d}.npy".format(self.cached_pt)), allow_pickle=True)
        self.target_samples = np.load(os.path.join(
          self.root, "target_samples_{:05d}.npy".format(self.cached_pt)), allow_pickle=True)




    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (x, y) where target is index of the target class.
        """
        index = self._get_rel_index(index)
        mel, mel_target = self.data_mels[index], self.target_mels[index]

        if self.transform is not None:
            mel = self.transform(mel)

        if self.target_transform is not None:
            mel_target = self.target_transform(mel_target)

        return mel, mel_target

    def __getinfo__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (x, y, aduiox, audioy) where target is index of the target class.
        """
        index = self._get_rel_index(index, samples=True)
        mel, mel_target, sample, sample_target = self.data_mels[index], self.target_mels[index], self.data_samples[index], self.target_samples[index]

        if self.transform is not None:
            mel = self.transform(mel)

        if self.target_transform is not None:
            mel_target = self.target_transform(mel_target)

        return mel, mel_target, sample, sample_target

    def _preloadinfo(self):
        self.data_samples = np.load(os.path.join(
          self.root, "raw_samples_{:05d}.npy".format(self.cached_pt)), allow_pickle=True)
        self.target_samples = np.load(os.path.join(
          self.root, "target_samples_{:05d}.npy".format(self.cached_pt)), allow_pickle=True)

    def _get_rel_index(self, index, samples=False):

        if self.cached_pt != index // self.chunk_size:
            self.cached_pt = int(index // self.chunk_size)
            self.data_mels = np.load(os.path.join(
                self.root, "raw_mel_{:05d}.npy".format(self.cached_pt)), allow_pickle=True)
            self.target_mels = np.load(os.path.join(
                self.root, "target_mel_{:05d}.npy".format(self.cached_pt)), allow_pickle=True)

            if (samples):
              self.data_samples = np.load(os.path.join(
                self.root, "raw_samples_{:05d}.npy".format(self.cached_pt)), allow_pickle=True)
              self.target_samples = np.load(os.path.join(
                self.root, "target_samples_{:05d}.npy".format(self.cached_pt)), allow_pickle=True)

        index = index % self.chunk_size
        return index

    def __len__(self):
        return self.num_samples-1

    def getRandomSamplesIndex(self, seed=42, validation=0.2):
        np.random.seed = seed
        train_idxs = np.array([])
        val_idxs = np.array([])
        val_chunck = range(self.__len__() // self.chunk_size)[(-int(validation*100)):]

        for i in range(self.__len__() // self.chunk_size):
            arr = np.arange(self.chunk_size) + i*self.chunk_size
            np.random.shuffle(arr)
            if i in val_chunck:
                val_idxs = np.concatenate((val_idxs, arr))
            else:
                train_idxs = np.concatenate((train_idxs, arr))
        return (train_idxs, val_idxs)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, "data.json"))

    def _read_info(self):
        info_path = os.path.join(
            self.root, "data.json")
        with open(info_path, "r") as f:
            d = json.load(f)
            self.sr = d['sample_rate']
            self.sr_target = d['sample_rate_target']
            self.num_samples = d['n_sample']
            self.chunk_size = d['sample_per_files']
