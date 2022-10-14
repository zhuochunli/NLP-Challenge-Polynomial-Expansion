# NLP-Challenge-Polynomial-Expansion
Implement a deep neural network model that learns to expand single variable polynomials.

[Problem Details](https://github.com/Chare7/NLP-Challenge-Polynomial-Expansion/blob/master/NLP%20Challenge.docx)

[Report](https://github.com/Chare7/NLP-Challenge-Polynomial-Expansion/blob/master/Report.pdf)

## Example
(7-3*z)*(-5*z-9)=15*z**2-8*z-63

(7-3*z)*(-5*z-9) is the factorized form

15*z**2-8*z-63 is the expanded form

## Dataset and model
Dataset: dataset.txt

Trained model: seq2seq.h5

Dataset statistics: data_summary.ipynb

## Reproduce
1. Proprocess the dataset and split it into train.txt, valid.txt and test.txt.(use --help for more options):
    ```
    python preprocess.py --dataset_path dataset.txt --seed 2022
    ```

2. Train the Seq2Seq model on train.txt and valid.txt:
    ```
    python train.py --hidden_dim 256 --batch_size 1024 --epochs 10 --learning_rate 0.005 --workers 2
    ```
3. Test the Seq2Seq model on test.txt:
    ```
    python test.py --batch_size 1024 --model_path seq2seq.h5 --test_path test.txt --workers 2
    ```

## Performance
Accuracy and loss on test data:

Test accuracy:  0.9859623312950134

Test loss:  0.038746245205402374

Training and Validation Accuracy curve:

![image](https://github.com/Chare7/NLP-Challenge-Polynomial-Expansion/blob/master/acc_history.png)

Training and Validation Loss curve:

![image](https://github.com/Chare7/NLP-Challenge-Polynomial-Expansion/blob/master/loss_history.png)

## References
https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
