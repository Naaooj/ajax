# TindHire model training

The model is currently based on [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased), a pretrained model on English language, used for natural language processing and interesting in a case of binary classification.

In order to train the model, the following classes are involved:
* ResumeDatasetBuilder
* ResumeDataset
* Model

And a script, `trainer.py`, responsible to run the training.