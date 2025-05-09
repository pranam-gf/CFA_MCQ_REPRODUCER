"""
Functions for evaluating LLM performance.
Includes metrics like accuracy, precision, recall, F1-score, and cosine similarity.
"""
import logging
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)

def evaluate_classification(results_data: list[dict]) -> dict:
    """
    Calculates classification metrics based on LLM answer correctness.
    """
    
    valid_results = [item for item in results_data if item.get('is_correct') is not None]

    default_metrics = {
        "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
        "confusion_matrix": [[0, 0], [0, 0]], 
        "classification_report": "No valid comparisons available.",
        "num_valid_comparisons": 0
    }

    if not valid_results:
        logger.warning("No entries with valid 'is_correct' flag found. Cannot calculate classification metrics.")
        return default_metrics

    
    num_correct = sum(1 for item in valid_results if item['is_correct'] is True)
    accuracy = num_correct / len(valid_results) if len(valid_results) > 0 else 0.0

    
    
    
    y_true = [str(item.get('correctAnswer', '')) for item in valid_results]
    y_pred = [str(item.get('LLM_answer', '')) for item in valid_results]
        
    filtered_y_true = [label for label in y_true if label]
    filtered_y_pred = [label for label in y_pred if label]

    all_labels_present = sorted(list(set(filtered_y_true + filtered_y_pred)))
    
    
    precision_avg = 0.0
    recall_avg = 0.0
    f1_avg = 0.0
    conf_matrix_list = [[0,0],[0,0]] 
    class_report_str = "Detailed report requires at least one common label between true and predicted sets after filtering."

    if not all_labels_present:
        logger.warning("No valid labels found in y_true or y_pred after filtering. Cannot compute detailed metrics beyond accuracy.")
        class_report_str = "No valid, non-empty labels found for detailed classification report."
    elif len(all_labels_present) == 1 and len(set(filtered_y_true)) == 1 and len(set(filtered_y_pred)) == 1 and filtered_y_true[0] == filtered_y_pred[0]:
        
        
        logger.info(f"Only one class ('{all_labels_present[0]}') present and predicted consistently. Accuracy is 1.0 or 0.0 based on correctness.")
        
        if accuracy == 1.0:
            precision_avg = 1.0
            recall_avg = 1.0
            f1_avg = 1.0
            
            
            
            if len(valid_results) > 0: 
                 conf_matrix_list = [[len(valid_results), 0], [0, 0]] if accuracy == 1.0 else [[0, len(valid_results)], [0,0]]


        class_report_str = f"Single class '{all_labels_present[0]}' consistently processed. Accuracy: {accuracy:.4f}"
    else:
        try:
            
            
            
            
            precision_avg = precision_score(y_true, y_pred, labels=all_labels_present, average='weighted', zero_division=0)
            recall_avg = recall_score(y_true, y_pred, labels=all_labels_present, average='weighted', zero_division=0)
            f1_avg = f1_score(y_true, y_pred, labels=all_labels_present, average='weighted', zero_division=0)
            
            
            cm = confusion_matrix(y_true, y_pred, labels=all_labels_present)
            conf_matrix_list = cm.tolist() 

            class_report_str = classification_report(y_true, y_pred, labels=all_labels_present, zero_division=0, target_names=all_labels_present)
            logger.info(f"Classification Report for {len(valid_results)} items:\n{class_report_str}")

        except ValueError as e:
            logger.error(f"ValueError during detailed classification metrics (sklearn): {e}. This might happen if y_true and y_pred have incompatible shapes or only one class after some internal sklearn processing. Labels: {all_labels_present}")
            class_report_str = f"Error in sklearn classification_report: {e}. Accuracy: {accuracy:.4f}"
        except Exception as e_sklearn: 
            logger.error(f"Unexpected error in sklearn metrics: {e_sklearn}", exc_info=True)
            class_report_str = f"Unexpected sklearn error. Accuracy: {accuracy:.4f}"

    calculated_metrics = {
        "accuracy": accuracy,
        "precision": precision_avg,
        "recall": recall_avg,
        "f1_score": f1_avg,
        "confusion_matrix": conf_matrix_list,
        "classification_report": class_report_str,
        "num_valid_comparisons": len(valid_results)
    }
    
    return calculated_metrics


