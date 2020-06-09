import torch
import numpy as np
import random
import timeit
from scipy.io import wavfile
import soundfile as sf
import json
import glob
import os

from segan_pytorch.segan.models import SEGAN
from segan_pytorch.segan.datasets import normalize_wave_minmax, pre_emphasize, argparse


class ArgParser(object):
    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)


def main(opts):
    assert opts.cfg_file is not None
    assert opts.test_files is not None
    assert opts.model is not None

    with open(opts.cfg_file, 'r') as cfg_f:
        args = ArgParser(json.load(cfg_f))
    segan = SEGAN(args)
    segan.G.load_pretrained(opts.model, True)
    segan.G.eval()

    wavs_list = glob.glob(os.path.join(opts.test_files, '*.wav'))

    print('Cleaning {} wav files'.format(len(wavs_list)))
    for index, wav_elem in enumerate(wavs_list, start=1):
        start_time = timeit.default_timer()

        file_name = os.path.basename(wav_elem)
        rate, wav = wavfile.read(wav_elem)
        wav = normalize_wave_minmax(wav)
        wav = pre_emphasize(wav, args.preemph)

        pwav = torch.FloatTensor(wav).view(1, 1, -1)
        g_wav, g_c = segan.generate(pwav)

        out_path = os.path.join(opts.synthesis_path, file_name)
        sf.write(out_path, g_wav, 16000)
        end_time = timeit.default_timer()
        print('Cleaned {}/{}: {} in {} s'.format(index, len(wavs_list), wav_elem, end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='segan_pytorch/ckpt_segan+/segan+_generator.ckpt')
    parser.add_argument('--test_files', type=str, nargs='+', default='small_noisy_testset/')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--synthesis_path', type=str, default='results')
    parser.add_argument('--cfg_file', type=str, default='segan_pytorch/ckpt_segan+/train.opts')

    opts = parser.parse_args()

    if not os.path.exists(opts.synthesis_path):
        os.makedirs(opts.synthesis_path)

    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)

    main(opts)
