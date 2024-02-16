## Step 1

```
pip install -r ./requirements.txt
```

Run `./generate_data.py` to automatically annotate the dataset based on wiki data. You can modify hyperparameters, paths, model names, and sizes in the file.

## Step 2

Run `generate_hd.py` to generate features needed for training the classifier based on the automatically annotated data.

## Step 3

Run `./train/train.py` to train the classifier.

## Step 4

Run `./detection_score.py` to evaluate the classifier's performance on the HELM dataset.