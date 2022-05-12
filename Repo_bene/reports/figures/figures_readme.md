## I. Confusion Matrix 

A confusion matrix gives a summary of the number of correct and incorrect predictions for the 10 categories.
On the x-axis we have the true label of the object and on the y-axis the ones predicted by our Yolov5. 
Aiming to predict correctly the labels, our goal is to get values close to 1 in the diagonal of the matrix.
This matrix gives us a better understanding of which categories are better predicted and those who aree more often misspredicted. 

Evaluating a model solely with this matrix would be misleading if the data set is not balanced. 

## II. Precision Curve

The precision is a measure of the number of predictions of a certain category properly predicted.
The precision measures the percentage of the labels classified as i out of all the labels classified as i.

precision = (nb. of labels properly classified as category i) / (nb. of labels predicted as category i)


## III. Recall Curve 

The recall measures the percentage of labels properly predicted for a category i, out of all the true category i labels. 

recall = (nb. of labels properly classified as category i) / (nb. of true labels that are of category i)

## IV. PR Curve

The PR curve has the precision on the y-axis and the recall on the x-axis. 
It is the plot of the positive predicted value against the true positive rate. 
It does not consider true negatives so it should not be used when we want to have specificity of the classifier, thus we would prioritize using other metrics to assess the performance of our model. It is of better use for binary classification problems.

A high area underneath this curve means that there is a high recall and a high precision. 
A high recall with a low precision means that a lot of objects are found and are being classified, but that the predictions are incorrect. 
A high precision and low recall means that there are few results but that the predictions are correct. 

Having a multi-label problem, we have that we create the curve for each of the categories vs. the other 9 grouped together, thus creating a binary problem.

## V. F1 curve 

## VI. Metrics 

#### 1) Box loss : 
How well the algorithm can locate the center of the object ; how well the predicted bounding box covers the object.

#### 2) Object loss : 
Measures whether or not the object exists in the predicted bounding box ; confidence of the object presence. 

#### 3) Classification loss : 
How well the model predicts the correct labels: classification loss.

#### 4) Précision : 
See precison curve. 

#### 5) Recall :
See recall curve. 







