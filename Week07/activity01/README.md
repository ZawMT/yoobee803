## Week 7 - Activity 1: confusion matrix
Create a confusion matrix for a healthcare system which developed based on using machine learning model to classify patients into two categories: Healthy and Sick based on medical test results and symptoms. The dataset contained 100 patient records collected from routine health screenings.
The dataset was divided into:
- 70 records for training
- 30 records for testing
After training the model, the testing phase was conducted using the 30 unseen patient records. The model performed like 3 false detections (incorrect predictions) during testing as follows; 3 records were misclassified:
- Two sick patients were predicted as healthy (False Negative).
- One healthy patients were predicted as sick (False Positive).
Share the description and figure of your confusion matrix here.
 
### Activity note
The confusion matrix should be as follows:
```
                                Prediction
                            Sick            Healthy
            Sick            13 (TP)          2 (FN)
Actual
            Healthy          1 (FP)          14 (TN)
```

Meaning:
TP (Sick) — predicted as Sick, actually Sick
TN (Healthy) — predicted as Healthy, and it was really Healthy
FP (Sick) — predicted as Sick, but it was actually Healthy
FN (Healthy) — predicted as Healthy, but actually Sick
