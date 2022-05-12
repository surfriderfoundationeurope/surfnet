# I. Confusion Matrix 

a confusion matrix gives a summary of the number of correct and incorrect predictions for the 10 categories.
On the x-axis we have the true label of the object and on the y-axis the ones predicted by our Yolov5. 
Aiming to predict correctly the labels, our goal is to get values close to 1 in the diagonal of the matrix.
This matrix gives us a better understanding of which categories are better predicted and those who aree more often misspredicted. 

Evaluating a model solely with this matrix would be misleading if the data set is not balanced. 

# II. Precision Curve



# III. Recall Curve 

# IV. PR Curve

The PR curve has the precision on the y-axis and the recall on the x-axis. 
It is the plot of the positive predicted value against the true positive rate. 
It does not consider true negatives so it should not be used when we want to have specificity of the classifier, thus we would prioritize using other metrics to assess the performance of our model. It is of better use ofr binary classification problems.


# II. F1 curve 





