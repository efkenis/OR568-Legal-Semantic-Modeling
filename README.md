# Instructions

1. The directory contains the data file in csv format (PTSD_claims_data.csv). You can actually even open this in excel if you like.
2. The requirements.txt file has all of the dependencies for the project which can be installed using 'pip install -r requirements.txt'. I'm sure there is also a way to do this in conda, but I'm not as familiar with it since I don't use it. Although, it likely that conda has all of this installed by default.
3. BVA_logistic_regression.py has all the steps of a typical NLP problem laid out. In it you will see the following,
   * Read in the data using pandas into a dataframe
   * Split the data into a training, validation, and test set (validation set is required to avoid overfitting data)
   * Set up a pipeline to vectorize, transform, and run a logistic regression (or classifcation) model. The for loop runs over several regularizer values C to find the one that leads to the highest predictive accuracy on the validation set. You can remove the logistic regression and use a different sklearn model relatively simply.
   * Using the C value determined above, the trained model is used to predict values for the testing set.
   * A confusion matrix and classifcation report is printed out.
