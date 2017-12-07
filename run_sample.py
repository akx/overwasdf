import argparse

import keras
import scipy.io.wavfile as wf

from overwasdf.sampling import do_sample_seq


ap = argparse.ArgumentParser()
ap.add_argument('--sample-rate', default=22050, type=int)
ap.add_argument('--output-len', default=1100, type=int)
ap.add_argument('--model', required=True)
ap.add_argument('--output', default='generated.wav')

if __name__ == '__main__':
    args = ap.parse_args()
    model = keras.models.load_model(args.model)
    _, vec_len, resolution = model.layers[0].input_shape
    seq, out_seq = do_sample_seq(
        model=model,
        seed_seq=None,
        vec_len=vec_len,
        n_samples=args.output_len,
        resolution=resolution,
    )
    print('Writing to %s' % args.output)
    wf.write(args.output, args.sample_rate, (out_seq - 0.5) * 2)
