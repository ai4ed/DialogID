from sklearn.metrics import *
import pandas as pd


def get_model_metrics(y_true, y_pred, show=False, tradeoff = 0.5):
    """
    Compute metrics to evaluate the model of a classification.

    Parameters:
        y_true: 1d array-like Ground truth (correct) labels.
        y_pred: Predicted labels, as returned by a classifier.
        show: Print result. Default value is False.

    Returns:
        report:Value of the metrics.
     
    Examples:
        ::

            {
                'Accuracy': 1,
                'Precision': 1,
                'Recall': 1,
                'F_measure': 1,
                'AUC_Value': 1
            }
    """
    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = -1
    y_pred = y_pred > tradeoff
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    if show:
        for name, value in zip(('Accuracy', 'Precision', 'Recall', 'F_meansure', 'AUC_Value'),
                               (accuracy, precision, recall, f1, auc)):
            print('{} : {:.4f}'.format(name, value))
    report = {'Accuracy': round(accuracy, 4),
              'Precision': round(precision, 4),
              'Recall': round(recall, 4),
              'F_meansure': round(f1, 4),
              'AUC_Value': round(auc, 4),
              }
    return report


def print_metris(report):
    columns = ['Accuracy','Precision','Recall','F_measure','AUC_Value']
    print('\t'.join([str(report[x]) for x in columns]))


def get_multi_class_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    return df_report


refit_map = {"acc": "Accuracy",
             "p": "Precision",
             "r": "Recall",
             "f1": "F_measure",
             "auc": "AUC_Value"}
