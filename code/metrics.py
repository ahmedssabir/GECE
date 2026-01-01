#!/usr/bin/env python
import numpy as np
import pandas as pd

def ece_score(py, y_test, num_bins=10, bin_output_file=None):
    py = np.array(py)
    y_test = np.array(y_test)
    
    
    if len(py.shape) == 1:  
        py = np.array([py, 1 - py]).T

    if py.shape[1] == 2:
        # For binary classification, treat the 0th column as the "positive" class
        py_index = np.zeros(py.shape[0], dtype="int") 
        y_test = np.abs(y_test - 1)
    else:
        py_index = np.argmax(py, axis=1) 

    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)

    acc, conf = np.zeros(num_bins), np.zeros(num_bins)
    Bm = np.zeros(num_bins)

    for m in range(num_bins):
        a, b = m / num_bins, (m + 1) / num_bins
        for i in range(py.shape[0]):
            if a < py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] /= Bm[m]
            conf[m] /= Bm[m]

    ece = np.sum(Bm * np.abs(acc - conf)) / np.sum(Bm) if np.sum(Bm) > 0 else 0

    if bin_output_file:
        with open(bin_output_file, 'w') as f:
            for m in range(num_bins):
                f.write(f"Bin {m}: Accuracy: {acc[m]}, Confidence: {conf[m]}, Size: {Bm[m]}\n")
            f.write(f"ECE: {ece}\n")
    
    return ece, acc.tolist(), conf.tolist(), Bm.tolist()

def compute_ice_macroce(probabilities, predicted_labels, true_labels):
    # Handle edge cases
    if len(probabilities) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    probabilities = np.array(probabilities)
    if np.any(np.isnan(probabilities)) or np.any(np.isinf(probabilities)):
        print("Warning: Found NaN or Inf in probabilities, replacing with 0.5")
        probabilities = np.nan_to_num(probabilities, nan=0.5, posinf=1.0, neginf=0.0)
    
    max_confidences = np.max(probabilities, axis=1)
    correctness = (predicted_labels == true_labels).astype(int)
    ice = np.mean(np.abs(correctness - max_confidences))
    correct_indices = correctness == 1
    incorrect_indices = correctness == 0
    ice_pos = np.mean(1 - max_confidences[correct_indices]) if np.sum(correct_indices) > 0 else 0
    ice_neg = np.mean(max_confidences[incorrect_indices]) if np.sum(incorrect_indices) > 0 else 0
    macroce = 0.5 * (ice_pos + ice_neg)
    return ice, macroce, ice_pos, ice_neg

def compute_binary_ece(probabilities, labels, num_bins=10):
    """
    Standard ECE for a binary classification problem wrt p(label=1).
    `probabilities` shape Nx2 => columns [p(label=0), p(label=1)].
    `labels` in {0,1}.
    """
    py = np.array(probabilities)
    y = np.array(labels)
    pos_conf = py[:, 1]

    bin_counts = np.zeros(num_bins)
    bin_acc_sum = np.zeros(num_bins)
    bin_conf_sum = np.zeros(num_bins)

    for i in range(len(pos_conf)):
        conf_val = pos_conf[i]
        true_label = abs(y[i] - 1)

        bin_index = int(conf_val * num_bins)
        if bin_index == num_bins:
            bin_index = num_bins - 1

        bin_counts[bin_index] += 1
        bin_conf_sum[bin_index] += conf_val
        if true_label == 1:
            bin_acc_sum[bin_index] += 1

    ece = 0.0
    total_count = len(pos_conf)
    for m in range(num_bins):
        if bin_counts[m] > 0:
            avg_acc = bin_acc_sum[m] / bin_counts[m]
            avg_conf = bin_conf_sum[m] / bin_counts[m]
            fraction = bin_counts[m] / total_count
            ece += fraction * abs(avg_acc - avg_conf)

    return ece

def compute_brier_score(probabilities, true_labels):
    # Handle edge cases
    if len(probabilities) == 0:
        return 0.0, np.array([])
    
    if len(probabilities.shape) == 1:
        probabilities = np.column_stack([probabilities, np.zeros_like(probabilities)])

    probabilities = np.array(probabilities)
    if np.any(np.isnan(probabilities)) or np.any(np.isinf(probabilities)):
        print("Warning: Found NaN or Inf in probabilities for Brier score, replacing with 0.5")
        probabilities = np.nan_to_num(probabilities, nan=0.5, posinf=1.0, neginf=0.0)
    
    positive_probs = probabilities[:, 1]  # Probability of the "positive" class
    squared_errors = (positive_probs - true_labels) ** 2
    brier_score = np.mean(squared_errors)
    return brier_score, squared_errors

def group_ece_by_bias_term(probabilities, true_labels, bias_terms, num_bins=10):
    probs = np.asarray(probabilities)
    y = np.asarray(true_labels)
    unique_groups = sorted({r for r in bias_terms if r is not None})
    if not unique_groups:
        return 0.0, 0.0, {}
    eces = {}
    for g in unique_groups:
        idx = [i for i, r in enumerate(bias_terms) if r == g]
        if len(idx) > 0:
            ece_g, *_ = ece_score(probs[idx], y[idx], num_bins=num_bins, bin_output_file=None)
            eces[g] = float(ece_g)
    vals = list(eces.values())
    group_ece = float(np.mean(vals)) if vals else 0.0
    bias_gap = float(np.max(vals) - np.min(vals)) if len(vals) > 1 else 0.0
    return group_ece, bias_gap, eces

def compute_group_ece_and_bias_gap(probabilities, labels, genders, num_bins=10):
    """
    Splits data into male/female group
    """

    def ece_for_group(group_probs, group_labels):
        if len(group_probs) == 0:
            print("Empty group encountered. Returning ECE=0.0")
            return 0.0
        return compute_binary_ece(group_probs, group_labels, num_bins=num_bins)

    male_probs = [probabilities[i] for i in range(len(genders)) if genders[i] == 'male']
    male_labels = [labels[i] for i in range(len(genders)) if genders[i] == 'male']

    female_probs = [probabilities[i] for i in range(len(genders)) if genders[i] == 'female']
    female_labels = [labels[i] for i in range(len(genders)) if genders[i] == 'female']

    print(f"\n--- Gender Debug Info ---")
    print(f"  Male samples: {len(male_probs)}")
    print(f"  Female samples: {len(female_probs)}")
    print(f"  Overall gender distribution: {dict(pd.Series(genders).value_counts(dropna=False))}")

    ece_male = ece_for_group(male_probs, male_labels)
    ece_female = ece_for_group(female_probs, female_labels)

    group_ece = 0.5 * (ece_male + ece_female)
    bias_gap = abs(ece_male - ece_female)
    return group_ece, bias_gap, ece_male, ece_female
