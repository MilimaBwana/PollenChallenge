# Pollen Classification
This project aims to classify different airbourne pollen with machine learning techniques.
Therefore, it uses python and Google's tensorflow framework.

## Dataset

[Link](https://iplab.dmi.unict.it/pollenclassificationchallenge/#traindata)

 | Latin      | English  | German    |
|------------|----------|-----------|
| Normal Corylus    | Normal Hazel    |  Normale Haselnuss |
| Anomalous Corylus    |  Anomalous Hazel | Anormale Haselnuss |
| Alnus      | Alder    | Erle      |
| Obruta 	| Debris | Trümmer|


### How to run

#### Preprocessing data
- First, you need to change DATA\_DIR\_4 in config/config.py to your local directory, in which the pollen data is stored. 
Each folder contains images of one specific class corresponding to the folder name.
- To create a tf record for training, validation and test, you need to run preprocessing/tf_record_writer.py with dataset_name='original\_4'
- If you want to upsample the dataset, you first need to run dataset\_manipulator.py with dataset_name='original\_4'. 

#### Run model
- Before running a model, you need to have created a tf\_record for training, evaluation and testing. The models can be found in the folder nn_models.
- Hyperparameters can be changed in the params dictionary, the model structure can be changed in the \_\_init\_\_() function.
- To run a model, you need the execute the module, in which the model is defined. 

#### Results
- Tracked performance metrics include recall, precision, loss and f1-score, which can be visualized via Tensorboard. 
- A confusion matrix is plotted for the performance of the test set, averaged over the number of iterations.
- Log files and checkpoints for a model are saved under nn\_models/logs/<model_name> by default. You can change it by adjusting LOG\_DIR and CHKPT\_DIR in config/config.py.

