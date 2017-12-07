import numpy as np
from tqdm import tqdm


def sample(preds, temperature=1.0, min_value=0, max_value=1):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    v = np.argmax(probas) / float(probas.shape[1])
    return v * (max_value - min_value) + min_value


def do_sample_seq(model, seed_seq, vec_len, n_samples, resolution):
    seq = (seed_seq or np.zeros(vec_len))
    seq_matrix = seq_to_matrix(seq, vec_len, resolution)
    out_seq = np.zeros(0)

    for i in tqdm(range(n_samples)):
        z = model.predict(seq_matrix.reshape((1, vec_len, resolution)))
        s = sample(z[0], 1.0)
        out_seq = np.append(out_seq, s)

        sample_ = int(s * (resolution - 1))
        seq_vec = np.zeros(resolution, dtype=bool)
        seq_vec[sample_] = True

        seq_matrix = np.vstack((seq_matrix, seq_vec))  # added generated note info
        seq_matrix = seq_matrix[1:]

    return (seq, out_seq)


def seq_to_matrix(seq, vec_len, resolution):
    seq_matrix = np.zeros((vec_len, resolution), dtype=bool)
    for i, s in enumerate(seq):
        sample_ = int(s * (resolution - 1))  # 0-255
        seq_matrix[i, sample_] = True
    return seq_matrix
