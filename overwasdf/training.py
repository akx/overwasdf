import numpy as np

from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint


def generate_training_data(input_vec, vec_len, resolution, step=5):
    # try to estimate next_sample (0 -255) based on 256 previous samples
    next_sample = []
    samples = []
    for j in range(0, input_vec.shape[0] - vec_len, step):
        seq = input_vec[j: j + vec_len + 1]
        seq_matrix = np.zeros((vec_len, resolution), dtype=bool)
        for i, s in enumerate(seq):
            sample_ = int(s * (resolution - 1))  # 0-255
            if i < vec_len:
                seq_matrix[i, sample_] = True
            else:
                seq_vec = np.zeros(resolution, dtype=bool)
                seq_vec[sample_] = True
                next_sample.append(seq_vec)
        samples.append(seq_matrix)
    samples = np.array(samples, dtype=bool)
    next_sample = np.array(next_sample, dtype=bool)
    return (samples, next_sample)


def do_train(model, checkpoint_name, samples, next_sample, epochs=100, batch_size=256):
    csv_logger = CSVLogger('{}_audio.log'.format(checkpoint_name))
    escb = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    checkpoint_template = '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % checkpoint_name
    checkpoint = ModelCheckpoint(filepath=checkpoint_template, monitor='val_loss', verbose=1, period=10)

    model.fit(
        samples,
        next_sample,
        shuffle=True,
        batch_size=batch_size,
        verbose=2,
        validation_split=0.1,
        epochs=epochs,
        callbacks=[csv_logger, escb, checkpoint]
    )
