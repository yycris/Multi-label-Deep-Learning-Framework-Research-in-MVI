# Multi-label-Deep-Learning-Framework-Research-in-MVI
Multi-label deep learning framework for preoperative microvascular invasion prediction and survival analysis in hepatocellular carcinoma.
## Requirements
* python3.11.4
* pytorch2.0.1+cu117
* tensorboard 2.8.0
## Usage
### 1.Dataset
* Arterial and venous phase liver CT 
* Lesser omentum adipose (LOA) CT
* Clinical information of patients with HCC.  
  
**PS:** The data **cannot be shared publicly** due to the privacy of individuals that participated in the study and because the data is intended for future research purposes.
### 2.Hyperparameter settings
* The hyperparameters of the training process are set in the file `$ config.py`.
* you can configure hyperparameters such as **batch_size**, **learning_rate**, and **epochs**.
### 3.Normalization
* Before training, image normalization is required to improve training efficiency and stability. 
* The mean and standard deviation of the training set can be obtained using the `$ standard.py` script.
### 4.Data augmentation
* Prior to training, data augmentation is applied to the training images to reduce the risk of overfitting.
* The data augmentation methods are defined in the file `$ utils.py`.
### 5.Train the Models
#### Model1--->CGAResNet18 trained using only dual-phase liver CT images.
* When instantiating the model1, the number of **input channels** is set to 3 and the number of **output classes** to 2. 
* **CrossEntropyLoss** is used for model optimization.
* The trained weight file used in the study, **cgaresnet18_tumor.pth**, is provided and can be found in the **/result/CGAResNet18 directory**.  
  
You need to train the model1 with the following commands:  
`$ python train.py`
#### Model2--->CGAResNet18 trained using dual-phase liver CT images with LOA CT images.
* When instantiating the model2, the number of **input channels** is set to 4 and the number of **output classes** to 2. 
* **CrossEntropyLoss** is used for model optimization.
* The trained weight file used in the study, **cgaresnet18_tumor_LOA.pth**, is provided and can be found in the **/result/CGAResNet18 directory**.  
  
You need to train the model2 with the following commands:  
`$ python train.py`
#### Model3--->CGAResNet18 trained using dual-phase liver CT images with LOA CT images and clinical multi-labels.
* When instantiating the model3, the number of **input channels** is set to 4 and the number of **output classes** to 4. 
* **BCEWithLogitsLoss** and **MSELoss** are used for model optimization.
* The trained weight file used in the study, **cgaresnet18_tumor_LOA_multiLabel.pth**, is provided and can be found in the **/result/CGAResNet18 directory**.  
  
You need to train the model3 with the following commands:  
`$ python train.py`
### 6.Predict MVI
If you want to see predictions for model1, model2 and model3, you should run the following file:  
`$ python predict.py`  
  
**PS:** It is important to configure the appropriate input channels, number of output classes, and load the corresponding weight files for each model.
### 7.Run tensorboard
`$ tensorboard --logdir=./logs/CGAResNet18`


