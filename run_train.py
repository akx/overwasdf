import glob
import os
import tempfile
import zipfile
import random
import time
import argparse
from overwasdf.model import create_model
from overwasdf.training import generate_training_data, do_train
from overwasdf.loading import load_vectors


def run_train(
    *,
    input_files,
    output_dir,
    sample_rate,
    vec_len,
    resolution,
    iterations=1000,
):
    vecs = load_vectors(audio_filenames=input_files, sample_rate=sample_rate)
    model = create_model(vec_len, resolution=resolution)
    vec_items = list(vecs.items())

    os.makedirs(output_dir, exist_ok=True)

    for iter in range(iterations):
        eid = 'au-r:%d-sr:%d-vl:%d-%s' % (
            resolution,
            sample_rate,
            vec_len,
            time.strftime('%Y%m%d-%H%M%s'),
        )
        filename, vec = random.choice(vec_items)
        print(eid, filename)
        samples, next_sample = generate_training_data(vec, vec_len=vec_len, resolution=resolution)
        do_train(model, os.path.join(output_dir, eid), samples, next_sample)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-rate', default=22050, type=int)
    ap.add_argument('--resolution', default=256, type=int)
    ap.add_argument('--vec-len', default=64, type=int)
    ap.add_argument('--iterations', default=1000, type=int)
    ap.add_argument('--output-dir', default='output')
    ap.add_argument('--input-zip')
    ap.add_argument('input_file', nargs='*')
    args = ap.parse_args()
    if args.input_zip:
        assert not args.input_file
        tempdir = tempfile.mkdtemp()
        with zipfile.ZipFile(args.input_zip) as zf:
            zf.extractall(tempdir)
        input_files = list(glob.glob(os.path.join(tempdir, '*.ogg')))
    else:
        input_files = list(args.input_file)

    run_train(
        input_files=input_files,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        vec_len=args.vec_len,
        resolution=args.resolution,
        iterations=args.iterations,
    )


if __name__ == '__main__':
    main()
