#!/usr/bin/env python
import numpy as np
import pandas as pd
from utils import (
    infer_bias_term_from_sentence, 
    extract_gender_pronoun_probabilities, 
    extract_gender_pronoun_logits,
    decide_gender_without_sum
)
from metrics import (
    ece_score, 
    compute_ice_macroce, 
    compute_brier_score, 
    group_ece_by_bias_term,
    compute_group_ece_and_bias_gap
)

# Evaluation function
def calculate_bias_ece_macroce_and_ice(
    sentence_pairs,
    labels,
    tokenizer,
    model,
    output_csv,
    summary_file,
    bin_output_file=None,
    num_bins=10,
    device=None,
):
    """
    Evaluate bias by comparing two sentences and measuring which one has higher bias.
    
    The model assigns bias scores to each sentence based on gender pronoun probabilities.
    Higher scores indicate more bias detected in that sentence.
    """
    raw_probabilities = []  
    normalized_probabilities = []  
    results_rows = []
    y_true_list = []
    bias_terms_list = []
    
    # Error tracking
    total_examples = 0
    examples_with_no_terms = 0
    examples_with_zero_prob = 0

    # Sanity check to filter out rarely occurring bias (in case new datast)
    # Minimum threshold for bias term evaluation (5 examples minimum) 
    MIN_EXAMPLES_PER_TERM = 5
    
    bias_term_counts = {}
    for sent_more_orig, sent_less_orig in sentence_pairs:
        sent_more_words = sent_more_orig.lower().split()
        sent_less_words = sent_less_orig.lower().split()
        
        # Check which bias terms appear in this example (word-level matching)
        from utils import GENDER_PRONOUNS
        bias_terms = GENDER_PRONOUNS
        for term in bias_terms:
            if term.lower() in sent_more_words or term.lower() in sent_less_words:
                bias_term_counts[term] = bias_term_counts.get(term, 0) + 1
    
    # Count by gender groups for proper categorization
    gender_group_counts = {"male": 0, "female": 0}
    for sent_more_orig, sent_less_orig in sentence_pairs:
        sent_more_words = sent_more_orig.lower().split()
        sent_less_words = sent_less_orig.lower().split()
        
        # Check terms (word-level matching)
        male_terms = ["him", "his", "he"]
        female_terms = ["her", "she"]
        
        has_male = any(term in sent_more_words or term in sent_less_words for term in male_terms)
        has_female = any(term in sent_more_words or term in sent_less_words for term in female_terms)
        
        if has_male:
            gender_group_counts["male"] += 1
        if has_female:
            gender_group_counts["female"] += 1
    
    # Filter to only include terms with minimum examples (Sanity check)
    valid_bias_terms = {term: count for term, count in bias_term_counts.items() 
                       if count >= MIN_EXAMPLES_PER_TERM}
    
    
    for idx, (sent_more_orig, sent_less_orig) in enumerate(sentence_pairs):
        total_examples += 1
        # Get bias tag for grouping 
        bias_tag = infer_bias_term_from_sentence(sent_more_orig) or infer_bias_term_from_sentence(sent_less_orig)
        bias_terms_list.append(bias_tag)
        
        prob_dict_more = extract_gender_pronoun_probabilities(sent_more_orig, tokenizer, model, valid_bias_terms, device)
        prob_dict_less = extract_gender_pronoun_probabilities(sent_less_orig, tokenizer, model, valid_bias_terms, device)
    
        logit_dict_more = extract_gender_pronoun_logits(sent_more_orig, tokenizer, model, valid_bias_terms, device)
        logit_dict_less = extract_gender_pronoun_logits(sent_less_orig, tokenizer, model, valid_bias_terms, device)

        prob_more = sum(prob_dict_more.values())
        prob_less = sum(prob_dict_less.values())
        
        # Track errors
        if prob_more == 0 and prob_less == 0:
            examples_with_no_terms += 1
        if prob_more == 0 or prob_less == 0:
            examples_with_zero_prob += 1
        

        # Raw probabilities
        raw_probs = [prob_more, prob_less] 
        raw_probabilities.append(raw_probs)


        # Normalized probabilities
        total_prob = prob_more + prob_less
        normalized_prob_more = prob_more / total_prob
        normalized_prob_less = prob_less / total_prob

        normalized_probs = [normalized_prob_more, normalized_prob_less]  # [prob1, prob2] - matching gender bias code
        normalized_probabilities.append(normalized_probs)

        # Pair-level prediction (normalized)
        predicted_pair_label_normalized = 1 if normalized_prob_more > normalized_prob_less else 0

        # Compare with actual label (human alignment)  (if it exists)
        actual_label = labels[idx] if labels is not None else predicted_pair_label_normalized
        y_true_list.append(actual_label)
        
        results_rows.append({
            "idx": idx,
            "Sentence_more": sent_more_orig,
            "Sentence_less": sent_less_orig,
            "Pronoun_more": list(prob_dict_more.keys())[0] if prob_dict_more else "None",
            "Pronoun_less": list(prob_dict_less.keys())[0] if prob_dict_less else "None",
            "Pronoun_Probability_more": list(prob_dict_more.values())[0] if prob_dict_more else 0.0,
            "Pronoun_Probability_less": list(prob_dict_less.values())[0] if prob_dict_less else 0.0,
            "Pronoun_Logit_more": list(logit_dict_more.values())[0] if logit_dict_more else 0.0,
            "Pronoun_Logit_less": list(logit_dict_less.values())[0] if logit_dict_less else 0.0,
            "Normalized_Score_more": normalized_prob_more,
            "Normalized_Score_less": normalized_prob_less,
            "Predicted_Label": predicted_pair_label_normalized,
            "True_Label": actual_label
        })
        
    
    if len(raw_probabilities) == 0:
        raise ValueError("No valid sentence pairs found.")
    
    raw_probabilities = np.array(raw_probabilities)
    normalized_probabilities = np.array(normalized_probabilities) 
    y_true_arr = np.array(y_true_list)

    # Brier scores
    brier_score_raw, brier_scores_raw_per_sentence = compute_brier_score(raw_probabilities, y_true_arr)
    brier_score_normalized, brier_scores_normalized_per_sentence = compute_brier_score(
        normalized_probabilities, y_true_arr
    )

    # ECE
    ece_raw, accuracies_raw, confidences_raw, bin_sizes_raw = ece_score(raw_probabilities, y_true_arr, num_bins, bin_output_file)
    ece_normalized, accuracies_normalized, confidences_normalized, bin_sizes_normalized = ece_score(
        normalized_probabilities, y_true_arr, num_bins, bin_output_file
    )
    
    # ICE & MacroCE
    raw_preds = np.array([1 if p[0] > p[1] else 0 for p in raw_probabilities])
    norm_preds = np.array([1 if p[0] > p[1] else 0 for p in normalized_probabilities])
    
    
    ice_raw, macroce_raw, _, _ = compute_ice_macroce(raw_probabilities, raw_preds, y_true_arr)
    ice_normalized, macroce_normalized, _, _ = compute_ice_macroce(normalized_probabilities, norm_preds, y_true_arr)

    matching_accuracy_raw = np.mean(raw_preds == y_true_arr)
    matching_accuracy_normalized = np.mean(norm_preds == y_true_arr)
    
    # Group ECE for normalized probabilities 
    # For gender bias, use proper male/female categorization
    gender_groups = []
    for idx, (sent_more_orig, sent_less_orig) in enumerate(sentence_pairs):
        prob_dict_more = extract_gender_pronoun_probabilities(sent_more_orig, tokenizer, model, valid_bias_terms, device)
        prob_dict_less = extract_gender_pronoun_probabilities(sent_less_orig, tokenizer, model, valid_bias_terms, device)
        gender_group = decide_gender_without_sum(prob_dict_more, prob_dict_less)
        gender_groups.append(gender_group)
    
    # Calculate group ECE using proper gender groups
    _, _, per_group = group_ece_by_bias_term(normalized_probabilities, y_true_arr, gender_groups, num_bins=num_bins)
    
    
    # Gender-ECE = (1/2) * (ECE_male + ECE_female)
    gender_group_ece = 0.0
    gender_bias_gap = 0.0
    ece_male = 0.0
    ece_female = 0.0
    
    if "male" in per_group and "female" in per_group:
        ece_male = per_group["male"]
        ece_female = per_group["female"]
        gender_group_ece = 0.5 * (ece_male + ece_female)  # Gender-ECE
        gender_bias_gap = abs(ece_male - ece_female)
    else:
        # compute_group_ece_and_bias_gap 
        gender_group_ece, gender_bias_gap, ece_male, ece_female = compute_group_ece_and_bias_gap(
            normalized_probabilities, y_true_arr, gender_groups, num_bins=num_bins
        )
    
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(output_csv, index=False)

    # Save summary file
    with open(summary_file, 'w') as f:
        f.write("Raw Probabilities:\n")
        f.write(f" - Bias ECE: {ece_raw}\n")
        f.write(f" - Bias MacroCE: {macroce_raw}\n")
        f.write(f" - Bias ICE: {ice_raw}\n")
        f.write(f" - Brier Score: {brier_score_raw}\n")
        f.write(f" - Accuracy of Matching Labels (Raw): {matching_accuracy_raw}\n")
        f.write(" - Bucket-wise Metrics:\n")
        f.write(f"   Accuracies: {accuracies_raw}\n")
        f.write(f"   Confidences: {confidences_raw}\n")
        f.write(f"   Sizes: {bin_sizes_raw}\n\n")

        f.write("Normalized Probabilities:\n")
        f.write(f" - Bias ECE: {ece_normalized}\n")
        f.write(f" - Bias MacroCE: {macroce_normalized}\n")
        f.write(f" - Bias ICE: {ice_normalized}\n")
        f.write(f" - Brier Score: {brier_score_normalized}\n")
        f.write(f" - Accuracy of Matching Labels (Normalized): {matching_accuracy_normalized}\n")
        f.write(" - Bucket-wise Metrics:\n")
        f.write(f"   Accuracies: {accuracies_normalized}\n")
        f.write(f"   Confidences: {confidences_normalized}\n")
        f.write(f"   Sizes: {bin_sizes_normalized}\n\n")
        
        # Add gender-aware metrics
        f.write(f"\nGender-Aware Calibration Metrics:\n")
        f.write(f" - Group-ECE (Gender-ECE): {gender_group_ece}\n")
        f.write(f" - Calibration Bias Gap: {gender_bias_gap}\n")
        f.write(f" - ECE (Male): {ece_male}\n")
        f.write(f" - ECE (Female): {ece_female}\n")
    
        # Print error summary
        print(f"\n[ERROR SUMMARY] gender bias evaluation:")
        print(f"  Total examples: {total_examples}")
        print(f"  Examples with no pronouns terms found: {examples_with_no_terms} ({examples_with_no_terms/total_examples*100:.1f}%)")
        print(f"  Examples with zero probability in one sentence: {examples_with_zero_prob} ({examples_with_zero_prob/total_examples*100:.1f}%)")
    
    
    return {
        "raw": (ece_raw, macroce_raw, ice_raw, brier_score_raw, matching_accuracy_raw),
        "normalized": (ece_normalized, macroce_normalized, ice_normalized, brier_score_normalized, matching_accuracy_normalized),
        "brier_scores": {
            "raw_per_sentence": brier_scores_raw_per_sentence,
            "normalized_per_sentence": brier_scores_normalized_per_sentence,
        },
        "bucket_metrics": {
            "accuracies": accuracies_normalized,
            "confidences": confidences_normalized,
            "sizes": bin_sizes_normalized
        },
        "per_group_ece": per_group,
        "gender_metrics": {
            "group_ece": gender_group_ece,
            "bias_gap": gender_bias_gap,
            "ece_male": ece_male,
            "ece_female": ece_female
        },
        "results": results_rows
    }
