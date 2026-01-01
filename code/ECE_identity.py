#!/usr/bin/env python
"""
ECE calculation for LGBTQ+ identity bias evaluation.
All labels are fixed to 1 (stereotypical) - no label mode.

Usage: python ECE_identity_simple.py --model_name gpt2 --dataset winoqueer.csv
"""

import argparse
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

# LGBTQ+ identity terms
IDENTITY_TERMS = {"straight", "gay", "lesbian", "transgender", "queer"}

def extract_identity_probabilities(sentence, tokenizer, model):
    """Extract probabilities for LGBTQ+ identity terms in a sentence."""
    model.eval()
    
    encoded = tokenizer(
        sentence,
        return_tensors='pt',
        return_offsets_mapping=True,
        add_special_tokens=False,
        max_length=512,
        truncation=True
    ).to(device)
    
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"][0]
    seq_len = input_ids.shape[1]
    
    if seq_len < 2:
        return {}
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
    
    shift_logits = logits[:, :-1, :]
    shift_probs = softmax(shift_logits, dim=-1).squeeze(0)
    
    bias_probs = {}
    token_substrings = []
    token_probs = []
    
    for i in range(seq_len):
        start_char, end_char = offsets[i].tolist()
        token_substring = sentence[start_char:end_char].strip()
        token_substrings.append(token_substring)
        
        if i > 0:
            token_id = input_ids[0, i].item()
            prob = shift_probs[i - 1, token_id].item()
            token_probs.append(float(prob))
        else:
            token_probs.append(0.0)
    
    # Check for identity terms
    for start_idx in range(len(token_substrings)):
        for end_idx in range(start_idx + 1, min(start_idx + 4, len(token_substrings) + 1)):
            combined_word = "".join(token_substrings[start_idx:end_idx])
            combined_word_lower = combined_word.lower()
            
            if combined_word_lower in IDENTITY_TERMS:
                word_prob = 0.0
                for token_idx in range(start_idx, end_idx):
                    if token_idx < len(token_probs):
                        word_prob = max(word_prob, token_probs[token_idx])
                
                for term in IDENTITY_TERMS:
                    if term.lower() == combined_word_lower:
                        bias_probs[term] = bias_probs.get(term, 0) + word_prob
                        break
    
    return bias_probs

def extract_identity_logits(sentence, tokenizer, model):
    """Extract raw logits for LGBTQ+ identity terms in a sentence."""
    model.eval()
    
    encoded = tokenizer(
        sentence,
        return_tensors='pt',
        return_offsets_mapping=True,
        add_special_tokens=False,
        max_length=512,
        truncation=True
    ).to(device)
    
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"][0]
    seq_len = input_ids.shape[1]
    
    if seq_len < 2:
        return {}
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
    
    shift_logits = logits[:, :-1, :].squeeze(0)
    
    bias_logits = {}
    
    for i in range(1, seq_len):
        start_char, end_char = offsets[i].tolist()
        actual_substring = sentence[start_char:end_char].strip()
        actual_substring_lower = actual_substring.lower()
        
        if actual_substring_lower in IDENTITY_TERMS:
            token_id = input_ids[0, i].item()
            raw_logit = shift_logits[i - 1, token_id].item()
            
            for term in IDENTITY_TERMS:
                if term.lower() == actual_substring_lower:
                    bias_logits[term] = max(bias_logits.get(term, float('-inf')), raw_logit)
                    break
    
    return bias_logits

def ece_score(probabilities, labels, num_bins=10):
    """Calculate Expected Calibration Error (ECE)."""
    py = np.array(probabilities)
    y_test = np.array(labels)
    
    if len(py.shape) == 1:
        py = np.array([py, 1 - py]).T
    
    if py.shape[1] == 2:
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
    
    return ece, acc.tolist(), conf.tolist(), Bm.tolist()

def main():
    parser = argparse.ArgumentParser(description="Simple ECE calculation for LGBTQ+ identity bias")
    parser.add_argument("--dataset", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--num_bins", type=int, default=10, help="Number of bins")
    parser.add_argument("--label", type=int, choices=[0, 1], default=1, help="Fix all labels to 0 (anti-stereotypical) or 1 (stereotypical)")
    args = parser.parse_args()
    
    # Load data
    data = pd.read_csv(args.dataset)
    sentence_pairs = data[['sent_more', 'sent_less']].values.tolist()
    
    # Fix all labels to specified value
    labels = [args.label] * len(sentence_pairs)
    label_type = "stereotypical" if args.label == 1 else "anti-stereotypical"
    print(f"All {len(sentence_pairs)} examples are {label_type} (label={args.label})")
    
    # Load model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            low_cpu_mem_usage=True
        )
    model.eval()
    
    # Calculate probabilities
    normalized_probabilities = []
    results_rows = []
    
    for idx, (sent_more, sent_less) in enumerate(sentence_pairs):
        prob_dict_more = extract_identity_probabilities(sent_more, tokenizer, model)
        prob_dict_less = extract_identity_probabilities(sent_less, tokenizer, model)
        
        logit_dict_more = extract_identity_logits(sent_more, tokenizer, model)
        logit_dict_less = extract_identity_logits(sent_less, tokenizer, model)
        
        prob_more = sum(prob_dict_more.values())
        prob_less = sum(prob_dict_less.values())
        
        total_prob = prob_more + prob_less
        if total_prob > 0:
            normalized_prob_more = prob_more / total_prob
            normalized_prob_less = prob_less / total_prob
        else:
            normalized_prob_more = 0.5
            normalized_prob_less = 0.5
        
        normalized_probabilities.append([normalized_prob_more, normalized_prob_less])
        
        # Pair-level prediction
        predicted_label = 1 if normalized_prob_more > normalized_prob_less else 0
        
        # Store results
        results_rows.append({
            "idx": idx,
            "Sentence_more": sent_more,
            "Sentence_less": sent_less,
            "LGBTQ_Term_more": list(prob_dict_more.keys())[0] if prob_dict_more else "None",
            "LGBTQ_Term_less": list(prob_dict_less.keys())[0] if prob_dict_less else "None",
            "Term_Probability_more": list(prob_dict_more.values())[0] if prob_dict_more else 0.0,
            "Term_Probability_less": list(prob_dict_less.values())[0] if prob_dict_less else 0.0,
            "Term_Logit_more": list(logit_dict_more.values())[0] if logit_dict_more else 0.0,
            "Term_Logit_less": list(logit_dict_less.values())[0] if logit_dict_less else 0.0,
            "Normalized_Score_more": normalized_prob_more,
            "Normalized_Score_less": normalized_prob_less,
            "Predicted_Label": predicted_label,
            "True_Label": args.label  # All fixed to specified label
        })
    
    # Calculate ECE
    normalized_probabilities = np.array(normalized_probabilities)
    ece, accuracies, confidences, bin_sizes = ece_score(
        normalized_probabilities, 
        labels, 
        num_bins=args.num_bins
    )
    
    # Print results
    dataset_name = os.path.basename(args.dataset).replace('.csv', '')
    print("\n" + "="*50)
    print(f"RESULTS - Identity Bias ECE ({dataset_name})")
    print("="*50)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Total examples: {len(sentence_pairs)}")
    print(f"\nECE Score: {ece:.4f}")
    print(f"\nBin Details:")
    for i in range(args.num_bins):
        print(f"  Bin {i}: Accuracy={accuracies[i]:.4f}, Confidence={confidences[i]:.4f}, Size={bin_sizes[i]:.0f}")
    print("="*50)
    
    # Create output folder with dataset name, model name, and label
    model_clean = args.model_name.split('/')[-1].replace('-', '_')
    dataset_clean = os.path.basename(args.dataset).replace('.csv', '')
    label_suffix = f"y_{args.label}"
    output_folder = f"{dataset_clean}_{label_suffix}_{model_clean}"
    os.makedirs(output_folder, exist_ok=True)
    
    # Save detailed results CSV (like original file)
    results_csv = os.path.join(output_folder, f"results_{dataset_clean}_{model_clean}.csv")
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(results_csv, index=False)
    
    # Save summary text file
    summary_file = os.path.join(output_folder, f"summary_{dataset_clean}_{model_clean}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Total examples: {len(sentence_pairs)}\n")
        f.write(f"ECE Score: {ece:.4f}\n\n")
        f.write(f"Bin Details:\n")
        for i in range(args.num_bins):
            f.write(f"  Bin {i}: Accuracy={accuracies[i]:.4f}, Confidence={confidences[i]:.4f}, Size={bin_sizes[i]:.0f}\n")
    
    # Save ECE bins CSV
    ece_csv = os.path.join(output_folder, f"ece_bins_{dataset_clean}_{model_clean}.csv")
    bins_data = {
        'Bin': list(range(args.num_bins)),
        'Accuracy': accuracies,
        'Confidence': confidences,
        'Size': bin_sizes
    }
    bins_df = pd.DataFrame(bins_data)
    bins_df.to_csv(ece_csv, index=False)
    
    print(f"\nAll results saved to folder: {output_folder}/")
    print(f"  - {os.path.basename(results_csv)}")
    print(f"  - {os.path.basename(summary_file)}")
    print(f"  - {os.path.basename(ece_csv)}")

if __name__ == "__main__":
    main()
