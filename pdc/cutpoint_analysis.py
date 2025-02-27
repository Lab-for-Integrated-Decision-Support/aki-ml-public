import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score, RocCurveDisplay, accuracy_score

def get_results_at_cutpoint(y_test, y_score, 
                           cutpoint=0.5,
                           cutpoint_error_allowance=0.02,
                           initial_pred_thresh=0.5,
                           max_iteration_count=100,
                           return_metrics=True,
                           debug_mode=False):
    positive_count_at_cutpoint = np.round((1 - cutpoint) * len(y_test))
    
    cutpoint_found = False
    current_pred_thresh = initial_pred_thresh
    pred_thresh_list = [0, 1]
    iteration = 1
    while cutpoint_found == False:
        y_pred = [1 if score > current_pred_thresh else 0 for score in y_score]
        y_positive_count = sum(y_pred)
        current_positive_count_error = (positive_count_at_cutpoint - y_positive_count) / 100
        # Check if predicted positive count within acceptable range of cutpoint count
        if abs(current_positive_count_error) <= cutpoint_error_allowance:
            cutpoint_found = True
        # Check if iteration count limit has been reached
        elif iteration >= max_iteration_count:
            cutpoint_found = True
            print('Max number of iterations reached - returning results early.')
        else:
            # If too few samples predicted positive, lower prediction threshold
            if current_positive_count_error > 0:
                candidate_thresholds = [x for x in pred_thresh_list if x < current_pred_thresh]
                new_pred_thresh = np.mean([max(candidate_thresholds), current_pred_thresh])
                pred_thresh_list = candidate_thresholds + [current_pred_thresh]
                current_pred_thresh = new_pred_thresh
            else:
                candidate_thresholds = [x for x in pred_thresh_list if x > current_pred_thresh]
                new_pred_thresh = np.mean([min(candidate_thresholds), current_pred_thresh])
                pred_thresh_list = candidate_thresholds + [current_pred_thresh]
                current_pred_thresh = new_pred_thresh

        iteration += 1
        if debug_mode:
            print('Iteration: %d' % iteration)

    cutpoint_results_dict = {
        'y_pred': y_pred,
        'pred_threshold': current_pred_thresh,
        'error_on_positive_count': current_positive_count_error,
        'y_pred_positive_count': sum(y_pred),
        'y_pred_negative_count': len(y_pred) - sum(y_pred),
        'actual_cutpoint': (len(y_pred) - sum(y_pred)) / len(y_pred)
    }

    if return_metrics:
        metric_dict = get_metrics_from_predictions(y_pred, y_test, cutpoint)
        cutpoint_results_dict['metrics'] = metric_dict

    return cutpoint_results_dict

def get_metrics_from_predictions(y_pred, y_true, cutpoint):
    metric_dict = {'cutpoint': cutpoint}
    metric_dict['accuracy'] = accuracy_score(y_true, y_pred)
    metric_dict['precision'] = precision_score(y_true, y_pred)
    metric_dict['tpr'] = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metric_dict['tnr'] = tn / (tn + fp)
    metric_dict['tp'] = tp
    metric_dict['tn'] = tn
    metric_dict['fp'] = fp
    metric_dict['fn'] = fn
    metric_dict['test_positive_count'] = tp + fp
    return metric_dict

def get_accuracy_at_cutpoint(y_test, y_score, 
                             cutpoint=0.5,
                             cutpoint_error_allowance=0.02,
                             initial_pred_thresh=0.5,
                             max_iteration_count = 1000):
    cutpoint_results = get_results_at_cutpoint(y_test,
                                               y_score,
                                               cutpoint,
                                               cutpoint_error_allowance,
                                               initial_pred_thresh,
                                               max_iteration_count)
    
    accuracy = cutpoint_results['metrics']['accuracy']
    
    return accuracy

def get_specificity_at_cutpoint(y_test, y_score, 
                             cutpoint=0.5,
                             cutpoint_error_allowance=0.02,
                             initial_pred_thresh=0.5,
                             max_iteration_count = 1000):
    cutpoint_results = get_results_at_cutpoint(y_test,
                                               y_score,
                                               cutpoint,
                                               cutpoint_error_allowance,
                                               initial_pred_thresh,
                                               max_iteration_count)
    
    specificity = cutpoint_results['metrics']['tnr']
    
    return specificity

def get_sensitivity_at_cutpoint(y_test, y_score, 
                             cutpoint=0.5,
                             cutpoint_error_allowance=0.02,
                             initial_pred_thresh=0.5,
                             max_iteration_count = 1000):
    cutpoint_results = get_results_at_cutpoint(y_test,
                                               y_score,
                                               cutpoint,
                                               cutpoint_error_allowance,
                                               initial_pred_thresh,
                                               max_iteration_count)
    
    sensitivity = cutpoint_results['metrics']['tpr']
    
    return sensitivity

def get_tpr_at_cutpoint(y_test, y_score, 
                         cutpoint=0.5,
                         cutpoint_error_allowance=0.02,
                         initial_pred_thresh=0.5,
                         max_iteration_count = 1000):
    tpr = get_sensitivity_at_cutpoint(y_test, y_score, 
                                     cutpoint=0.5,
                                     cutpoint_error_allowance=0.02,
                                     initial_pred_thresh=0.5,
                                     max_iteration_count = 100)
    return tpr

def get_tnr_at_cutpoint(y_test, y_score, 
                         cutpoint=0.5,
                         cutpoint_error_allowance=0.02,
                         initial_pred_thresh=0.5,
                         max_iteration_count = 1000):
    tnr = get_specificity_at_cutpoint(y_test, y_score, 
                                     cutpoint=0.5,
                                     cutpoint_error_allowance=0.02,
                                     initial_pred_thresh=0.5,
                                     max_iteration_count = 100)
    return tnr

def get_precision_at_cutpoint(y_test, y_score, 
                             cutpoint=0.5,
                             cutpoint_error_allowance=0.02,
                             initial_pred_thresh=0.5,
                             max_iteration_count = 1000):
    cutpoint_results = get_results_at_cutpoint(y_test,
                                               y_score,
                                               cutpoint,
                                               cutpoint_error_allowance,
                                               initial_pred_thresh,
                                               max_iteration_count)
    
    accuracy = cutpoint_results['metrics']['precision']
    
    return accuracy
