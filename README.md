# Machine_Learning-Classification_Regression_and_Classifier_interpretability
1. **Classification with Hyperparameter Search** : The idea here is to train and evaluate 8 classification methods across 10 classification datasets. 

2. **Regression with Hyperparameters Search** : The idea here is to train and evaluate 7 regression methods across 10 regression datasets. 

3. **Classifier interpretability** : load and train models on standard computer vision dataset called CIFAR-10 and train a convolutional neural network using PyTorch to classify images in the dataset; train a decision tree to classify images in the dataset; and try to interpret the CNN using the 'activation maximization' technique. 

4. **Novelty component** : Try to introduce a novel aspect to your analysis of classifiers and regressors or to your investigation of interpretability.

There are 3 notebooks:

    1. Classification 
    2. Regression
    3. CIFAR

Important Instructions for Classification and Regression

    Run Files in one by one or in batch.

Datasets have been given a unique ID :-  ex. Wine Quality - RP_1 | Adult - CP_7

Section 4. Code Cell will train all models one by one on all the dataset with a call of classification() or regress() method.

     • It will train DEFAULT SKLEARN MODELS

     • Extensive Analysis Result can be found in RegressionAnalysis.XLS and ClassifiationAnalysis.XLS
 
For **HyperParameter Search** - USE hs=True along with other parameters.
	
    • Example: regress(RP_4_X_train, RP_4_X_test, RP_4_y_train, RP_4_y_test, hs=True)

    • HyperParameter Search option will give best estimators using both GridSearchCV and RandomizedSearchCV for each models.

    • Some Results of the Exhaustive Hyperparameter Search can be found in RP_Data or CP_Data folders under Hyperparameter Search Results for all the datasets.

    • For the Best Result find the parameters we found from the Config.XLS File and follow the instructions in the last cell block.
	Example: ABR(RP_1_X_train, RP_1_X_test, RP_1_y_train, RP_1_y_test, hs=False, parity=0, n_estimators=150, learning_rate=0.1, loss='exponential')
	Values like n_estimators, learning_rate and loss in this case for RP_1 Dataset (Wine Quality) are taken from Config.XLS

 • In Regression, parity = 1 & 2 is used to handle Merck Dataset. parity = 3 will give MSE result. Otherwise it will give R^2 Score.

 • In Classifiation, parity handles different evaluation metrics. 

 • For CIFAR, variable data_dir contains the path to raw cifar10 dataset. Currently it looks like this data_dir = 'cifar10/' 

**Novelty Component:**

    • NLP on WhatsApp and the text to train is in RomeoJulietExtract.txt 
    • Ridge Regression is in Regression Notebook
