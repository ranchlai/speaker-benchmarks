import argparse
import glob
import pickle
import random

import librosa
import numpy as np
import torch
import tqdm

from metrics import compute_eer
from speaker_model import SpeakerSB, SpeakerSV  # noqa


def test(spk2feats):

    keys = spk2feats.keys()
    keys = list(keys)
    keys.sort()

    n_iter = 1000
    scores = []
    labels = []
    for _ in range(n_iter):
        k1 = random.choice(keys)
        k2 = random.choice(keys)
        while k1 == k2:
            k2 = random.choice(keys)

        X = np.concatenate(spk2feats[k1])
        Y = np.concatenate(spk2feats[k2])

        X /= (np.sum(X**2, 1, keepdims=True)**0.5)
        Y /= (np.sum(Y**2, 1, keepdims=True)**0.5)
        inner_dist = np.dot(X, X.T)
        inner_dist = inner_dist[np.where(
            ~np.eye(inner_dist.shape[0], dtype=bool))]
        scores += [inner_dist]
        labels += [np.ones_like(inner_dist)]

        inner_dist = np.dot(Y, Y.T)
        inner_dist = inner_dist[np.where(
            ~np.eye(inner_dist.shape[0], dtype=bool))]
        scores += [inner_dist]
        labels += [np.ones_like(inner_dist)]

        inter_dist = np.dot(X, Y.T).reshape(-1)
        scores += [inter_dist]
        labels += [np.zeros_like(inter_dist)]

    scores = np.concatenate(scores)
    labels = np.concatenate(labels)

    out = compute_eer(scores, labels)
    print(f'eer: {out.eer}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',
                        '--folder',
                        type=str,
                        required=False,
                        default='./data/aishell3/',
                        help='aishell3 data folder')

    parser.add_argument('-n',
                        '--n_files_per_speaker',
                        type=int,
                        required=False,
                        default=32)

    parser.add_argument('-s', '--seed', type=int, required=False, default=100)
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        required=False,
                        default='SpeakerSB')
    parser.add_argument('-u',
                        '--duration',
                        type=int,
                        required=False,
                        default=3)

    parser.add_argument(
        '-d',
        '--device',
        default='cuda',
        choices=['cpu', 'cuda'],
        help='Select which device to train model, defaults to gpu.')

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f'Using model {args.model}')
    model = eval(args.model)()

    wav_files = glob.glob(f'{args.folder}/wavs/*.wav')
    wav_files.sort()
    print(f'{len(wav_files)} found')

    spks = [f.split('/')[-1][:7] for f in wav_files]
    spk_set = list(set(spks))
    spk_set = spk_set
    print(f'{len(spk_set)} speakers found')

    spk2files = {s: [] for s in spk_set}
    for f in wav_files:
        spk = f.split('/')[-1][:7]
        spk2files[spk] += [f]

    spk2feats = {s: [] for s in spk_set}
    for spk in tqdm.tqdm(spk_set):
        for file in tqdm.tqdm(spk2files[spk][:args.n_files_per_speaker]):
            signal, fs = librosa.load(file,
                                      sr=16000,
                                      duration=args.duration,
                                      res_type='kaiser_fast')
            signal = torch.tensor(signal[:16000 * args.duration])[None, :].to(
                args.device)
            feat = model(signal)
            spk2feats[spk] += [feat.cpu().numpy()]

    with open(f'spk2feats_{args.model}.pkl', 'wb') as fp:
        pickle.dump(spk2feats, fp)

    test(spk2feats)
