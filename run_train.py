import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
import random
import time
import argparse
from overwasdf.model import create_model
from overwasdf.training import generate_training_data, do_train
from overwasdf.loading import load_vectors

ap = argparse.ArgumentParser()
ap.add_argument('--sample-rate', default=22050, type=int)
ap.add_argument('--resolution', default=256, type=int)
ap.add_argument('--vec-len', default=64, type=int)
ap.add_argument('--iterations', default=1000, type=int)
ap.add_argument('--output-dir', default='output')
ap.add_argument('input_file', nargs='+')
args = ap.parse_args()

vecs = load_vectors(audio_filenames=args.input_file, sample_rate=args.sample_rate)
model = create_model(args.vec_len, resolution=args.resolution)
vec_items = list(vecs.items())

os.makedirs(args.output_dir, exist_ok=True)

for iter in range(args.iterations):
    eid = 'au-r:%d-sr:%d-vl:%d-%s' % (
        args.resolution,
        args.sample_rate,
        args.vec_len,
        time.strftime('%Y%m%d-%H%M%s'),
    )
    filename, vec = random.choice(vec_items)
    print(eid, filename)
    samples, next_sample = generate_training_data(vec, vec_len=args.vec_len, resolution=args.resolution)
    do_train(model, os.path.join(args.output_dir, eid), samples, next_sample)
