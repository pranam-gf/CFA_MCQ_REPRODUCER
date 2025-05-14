import logging
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    classification_report
)
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def evaluate_classification(results_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates classification performance based on LLM results.

    Args:
        results_data: A list of dictionaries, where each dictionary contains
                      at least 'correct_answer' and 'LLM_answer'.

    Returns:
        A dictionary containing classification metrics:
        'accuracy', 'precision', 'recall', 'f1_score', 
        'confusion_matrix', 'classification_report' (string).
    """
    if not results_data:
        logger.warning("No results data provided for classification evaluation.")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "confusion_matrix": [],
            "classification_report": "No data provided.",
            "labels": []
        }

    y_true = []
    y_pred = []
    valid_labels = set()

    for item in results_data:
        true_label = item.get('correctAnswer')
        pred_label = item.get('LLM_answer')
        if isinstance(true_label, str) and true_label and "PLACEHOLDER" not in true_label:
            true_label_upper = true_label.strip().upper()
            valid_labels.add(true_label_upper)  
            if not isinstance(pred_label, str) or not pred_label or pred_label.startswith("ERROR") or pred_label == "PARSE_FAIL":
                 
                 pred_label_processed = "INVALID_PRED" 
            else:
                 pred_label_processed = pred_label.strip().upper()

            y_true.append(true_label_upper)
            y_pred.append(pred_label_processed)
        else:
             logger.debug(f"Skipping item due to invalid/missing true label: {item}")

    if not y_true:
        logger.warning("No valid pairs of true and predicted labels found for evaluation.")
        return {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
            "confusion_matrix": [], "classification_report": "No valid labels.", "labels": []
        }
    all_unique_labels = sorted(list(valid_labels.union(set(y_pred))))
    
    if "INVALID_PRED" in set(y_pred) and "INVALID_PRED" not in all_unique_labels:
        all_unique_labels.append("INVALID_PRED")
        all_unique_labels = sorted(all_unique_labels) 
    try:
        accuracy = accuracy_score(y_true, y_pred)   
        report_labels = sorted(list(valid_labels))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', labels=report_labels, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=all_unique_labels)
        report_str = classification_report(
            y_true, y_pred, labels=all_unique_labels, zero_division=0, digits=4 
        )
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(), 
            "classification_report": report_str,
            "labels": all_unique_labels 
        }
        logger.info(f"Classification evaluation successful. Accuracy: {accuracy:.4f}")

    except Exception as e:
        logger.error(f"Error during classification evaluation: {e}", exc_info=True)
        metrics = {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
            "confusion_matrix": [], "classification_report": f"Error: {e}", "labels": all_unique_labels
        }
    return metrics
