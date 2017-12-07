from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input
from keras.optimizers import RMSprop


def create_model(samples, resolution=256, latent_dim=128, dropout=0.4):
    inputs = Input(shape=(samples, resolution))
    x = LSTM(latent_dim, return_sequences=True)(inputs)
    x = Dropout(dropout)(x)
    x = LSTM(latent_dim)(x)
    x = Dropout(dropout)(x)
    output = Dense(resolution, activation='softmax')(x)
    model = Model(inputs, output)

    # optimizer = Adam(lr=0.005)
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model
