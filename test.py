from keras.models import load_model
import argparse
from preprocess import Preprocess


def test(data_process, model_path, test_path, batch_size=512, workers_num=1):
    model = load_model(model_path)
    test_encoder_input, test_decoder_input_data, test_decoder_target_data = data_process.generate_data(test_path)
    loss, accuracy = model.evaluate([test_encoder_input, test_decoder_input_data], test_decoder_target_data,
                                    batch_size=batch_size,
                                    workers=workers_num)

    return loss, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Testing data: ")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model_path", type=str, default="seq2seq.h5")
    parser.add_argument("--dataset_path", type=str, default="dataset.txt")
    parser.add_argument("--test_path", type=str, default="test.txt")
    args = parser.parse_args()

    data_process = Preprocess(args.dataset_path)
    data_process.build_vocab()
    loss, acc = test(data_process, args.model_path, args.test_path, args.batch_size, args.workers)
    print('Test accuracy: ', acc, 'Test loss: ', loss)
