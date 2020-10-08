# Pollen Challenge
In this repository, we present a approach based on pre-trained convolutional
neural networks (CNNs) for the 2020 Pollen Challenge.

## Installation
```bash
pip install -r requirements_classification.txt
```
Alternatively, installing the following package should install the other needed packages, too:
- matplotlib==3.3.0
- opencv-python==4.3.0.36
- pytest==6.0.0
<<<<<<< HEAD
- notebook==6.1.3
- scikit-learn==0.23.1
- tensorflow(-gpu)==2.3.0
 
## Dataset

[Link](https://iplab.dmi.unict.it/pollenclassificationchallenge/#traindata)

 | Latin      | English  | German    |
|------------|----------|-----------|
| Normal Corylus    | Normal Hazel    |  Normale Haselnuss |
| Anomalous Corylus    |  Anomalous Hazel | Anormale Haselnuss |
| Alnus      | Alder    | Erle      |
| Obruta 	| Debris | Trümmer|

 In the code, the dataset is called orginal_4. 
 

### How to run 

#### Preprocessing data

-  The dataset shipped without labels for the testset from the website. If you have a json file containing the labels for the test dataset, copy it to your DATA_DIR_4 and run
```bash
python ./preprocessing/italian_data/move_test_data.py 
```
- You need a class name file where the names of the used pollen are listed line by line. It can be found [here](original_4.names). It is recommended to store the class name file in your DATA_DIR. Alternatively the path to the class name file can be changed in dataset/dataset_config.py

#### Create tf records
```bash
python ./preprocessing/tf_record_writer.py  --dataset_name original_4  
```

#### Data Upsampling/Downsampling
- If you want to upsample your data, run 
```bash
python ./preprocessing/dataset_manipulator.py  --dataset_name original_4 --max_fraction 0.1 --max_multiplication 5  #for Catania4
```
- max_multiplication and max_fraction are parameters to control the number of upsampled data per class.
- Create a tf record for upsampled data is done by running
```bash
python ./preprocessing/tf_record_writer.py  --dataset_name upsample_4
```

#### Run model
- Before running a model, you need to have created a tf\_record for training, evaluation and testing. The models can be found in the folder nn_models.
- Hyperparameters can be changed by command line arguments, the model structure can be changed in the \_\_init\_\_() function.
- To run a model, you need the execute the module, in which the model is defined, e.g:
```bash
python -u ./nn_models/transfer_learning_cnn.py --name TransferLearningCNN --dataset_name original_4 --input_shape 84 84 3 --backbone densenet --freeze 0.3 --learning_rate 1e-4 --lr_strategy PlateauDecay --dropout 0.5 --batch_size 64 --weighted_gradients FocalLoss --regularized_loss L2 --normalized_weights None -augmentation --augmentation_techniques flip_left_right flip_up_down crop rotate --epochs 50 --iterations 1
```
- An overview of all command line arguments can be found in nn_models/transfer_learning_cnn.py
- Currently, running a transfer learning model and a self-build CNN is possible


#### Results
- Tracked performance metrics include recall, precision, loss and f1-score, which can be visualized via Tensorboard. 
- A confusion matrix is plotted for the performance of the test set, averaged over the number of iterations.
- Log files and checkpoints for a model are saved under nn_models/logs/<model_name> by default. You can change it by adjusting LOG_DIR and CHKPT_DIR in config/config.py.
