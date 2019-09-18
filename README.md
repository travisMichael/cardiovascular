# Cardiovascular and Financial Loan Data Analysis

This project explores several supervised learning algorithms to classify cardiovascular disease. The project also 
explores several supervised learning algorithms to classify whether a financial loan is good or bad for the company
giving out the loan.

The project consists of multiple python programs that require python 3.6.8 or above. The project contains the following 
python dependencies (numpy, sklearn, matplotlib, and pickle).

The project can be cloned by running the following command:

https://github.com/travisMichael/cardiovascular.git

Before running any of the programs in this project, download the datasets from the following locations. Both datasets
are owned by Kaggle, so you must be a Kaggle member to download them. Signing up for Kaggle is free. Make sure to 
place both datasets inside the parent directory of this project.

Once both datasets have been placed inside the project, then the pre-processing programs can be run for each dataset. 
To run the pre-processing programs, run the following commands:

python dataPreProcessor.py cardio

python dataPreProcessor.py loan


Once the pre-processing programs have completed, then the programs to train the models can be run. To train the models,
run the following commands:

python train.py cardio all False

python train.py loan all False

If you would like to re-generate the learning curve plots for each model, run the following commands:

python train.py cardio all True

python train.py loan all True


Once the models have been trained, they will be ready for testing. The testing program generates precision-recall plots
the models prediction scores on the test data. The program also outputs the f1 score for each model, as well the amount
of time the model needed to make its predictions. To run the testing programs, run the following commands:

python test.py cardio all

python test.py loan all


All plots are saved under the plot directory.