import pickle
import numpy as np
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import argparse
from preprocess import Preprocess
from matplotlib import pyplot as plt


# build Seq2Seq model
def model(vocab_size=35, hid_dim=256, l_rate=0.005):
    # encoder part
    encoder_inputs = Input(shape=(None,))
    x = Embedding(vocab_size, hid_dim)(encoder_inputs)
    x, state_h, state_c = LSTM(hid_dim, return_state=True)(x)
    encoder_states = [state_h, state_c]

    # decoder part, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,))
    x = Embedding(vocab_size, hid_dim)(decoder_inputs)
    x, _, _ = LSTM(hid_dim, return_sequences=True, return_state=True)(x, initial_state=encoder_states)
    decoder_outputs = Dense(vocab_size, activation='softmax')(x)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile and summary
    opt = keras.optimizers.RMSprop(learning_rate=l_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # seq2seq.summary()
    return model


def train(data_process, model, train_path, valid_path, batch_size=1024, epochs=10, workers_num=1):
    train_encoder_input, train_decoder_input_data, train_decoder_target_data = data_process.generate_data(train_path)
    valid_encoder_input, valid_decoder_input_data, valid_decoder_target_data = data_process.generate_data(valid_path)
    history = model.fit([train_encoder_input, train_decoder_input_data], train_decoder_target_data,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=([valid_encoder_input, valid_decoder_input_data], valid_decoder_target_data),
                        workers=workers_num)

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training data: ")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dataset_path", type=str, default="dataset.txt")
    parser.add_argument("--train_path", type=str, default="train.txt")
    parser.add_argument("--valid_path", type=str, default="valid.txt")
    args = parser.parse_args()

    # data process, build vocabulary table
    data_process = Preprocess(args.dataset_path)
    data_process.build_vocab()

    # compile model and train
    seq2seq = model(hid_dim=args.hidden_dim, l_rate=args.learning_rate)
    train_history = train(data_process, seq2seq, args.train_path, args.valid_path, args.batch_size, args.epochs,
                          args.workers)

    # save model and training history
    seq2seq.save('seq2seq.h5')
    print("Saved model to disk!")
    with open('seq2seq_trainHistoryDict.txt', 'wb') as file_pi:
        pickle.dump(train_history.history, file_pi)
    print("Saved epochs history to disk!")