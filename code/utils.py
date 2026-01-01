#!/usr/bin/env python
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
import re

# Gender pronouns to detect for bias evaluation
GENDER_PRONOUNS = {"her", "she", "him", "his", "he"}

def infer_bias_term_from_sentence(sentence: str):
    """
    Find the first gender pronoun in a sentence.
        "The doctor said he was confident" -> "he"
    """
    s = (sentence or "").lower()
    words = s.split()
    bias_terms = GENDER_PRONOUNS
    for term in bias_terms:
        if term in words:
            return term
    return None

def extract_gender_pronoun_probabilities(sentence: str, tokenizer, model, valid_bias_terms=None, device=None):
    """
    Returns a dictionary of bias term probabilities for words that match
    terms in the corresponding bias terms set EXACTLY in the original text.
    Example return: { 'he': 0.123, 'she': 0.045 } for gender
    """
    model.eval()

    # Tokenize with offsets to get exact substring positions
    encoded = tokenizer(
        sentence,
        return_tensors='pt',
        return_offsets_mapping=True,   # crucial to retrieve character positions
        add_special_tokens=False,
        max_length=512,  
        truncation=True  
    ).to(device)

    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"][0]  # shape: (seq_len, 2)
    seq_len = input_ids.shape[1]

    if seq_len < 2:
        return {}

    # Forward pass to get logits
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # shape: [1, seq_len, vocab_size]

    # Next-token probabilities come from the previous position
    shift_logits = logits[:, :-1, :]  # shape: [1, seq_len-1, vocab_size]
    shift_probs = softmax(shift_logits, dim=-1).squeeze(0)  # shape: [seq_len-1, vocab_size]

    bias_probs = {}

    # Get bias terms set
    if valid_bias_terms is not None:
        bias_terms = set(valid_bias_terms.keys())
    else:
        bias_terms = GENDER_PRONOUNS  # Always use gender terms
    
    # First, collect all token substrings and their probabilities
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
    
    
    # Now try to reconstruct words by combining consecutive tokens
    for start_idx in range(len(token_substrings)):
        for end_idx in range(start_idx + 1, min(start_idx + 4, len(token_substrings) + 1)):  # Try up to 3-token combinations
            # Combine tokens from start_idx to end_idx-1
            combined_word = "".join(token_substrings[start_idx:end_idx])
            combined_word_lower = combined_word.lower()
            
            
            # Check if this combined word matches any bias term
            if combined_word_lower in bias_terms:
                # Calculate the probability for this word
                # For multi-token words, use the maximum probability (not sum)
                word_prob = 0.0
                for token_idx in range(start_idx, end_idx):
                    if token_idx < len(token_probs):
                        word_prob = max(word_prob, token_probs[token_idx])
                
                # Get the original case from the sentence
                word_start_char = offsets[start_idx][0].item()
                word_end_char = offsets[end_idx-1][1].item()
                original_case_word = sentence[word_start_char:word_end_char].strip()
                
                # Store the probability (use the matched bias term as key for consistency)
                # Find the exact matching bias term (case-insensitive)
                matched_term = None
                for term in bias_terms:
                    if term.lower() == combined_word_lower:
                        matched_term = term
                        break
                
                if matched_term:
                    bias_probs[matched_term] = bias_probs.get(matched_term, 0) + word_prob
                else:
                    # Fallback to original case
                    bias_probs[original_case_word] = bias_probs.get(original_case_word, 0) + word_prob
    

    return bias_probs

def extract_gender_pronoun_logits(sentence: str, tokenizer, model, valid_bias_terms=None, device=None):
    """
    Extract raw logits (before softmax) for gender pronouns in a sentence.
    dict: Raw logits for each pronoun found (e.g., {'he': 2.3, 'she': 0.8})
    """
    model.eval()

    # Tokenize with offsets to get exact substring positions
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

    # Forward pass to get logits
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  

    # Next-token logits come from the previous position
    shift_logits = logits[:, :-1, :]  # shape: [1, seq_len-1, vocab_size]
    shift_logits = shift_logits.squeeze(0)  # shape: [seq_len-1, vocab_size]

    pronoun_logits = {}

    # Get the appropriate bias terms set
    if valid_bias_terms is not None:
        bias_terms = set(valid_bias_terms.keys())
    else:
        bias_terms = GENDER_PRONOUNS

    # Check individual tokens for pronoun matches
    for i in range(1, seq_len):
        start_char, end_char = offsets[i].tolist()
        actual_substring = sentence[start_char:end_char].strip()
        actual_substring_lower = actual_substring.lower()
        
        # If the substring matches a bias term, get its raw logit
        if actual_substring_lower in bias_terms:
            token_id = input_ids[0, i].item()
            raw_logit = shift_logits[i - 1, token_id].item()
            
            # Find the exact matching bias term
            matched_term = None
            for term in bias_terms:
                if term.lower() == actual_substring_lower:
                    matched_term = term
                    break
            
            if matched_term:
                pronoun_logits[matched_term] = max(pronoun_logits.get(matched_term, float('-inf')), raw_logit)
            else:
                pronoun_logits[actual_substring] = max(pronoun_logits.get(actual_substring, float('-inf')), raw_logit)

    return pronoun_logits

# Load sentence pairs and human alignment (labels) from CSV
def load_sentence_pairs_from_csv(input_csv, use_actual_labels=False):
    data = pd.read_csv(input_csv)
    sentence_pairs = data[['sent_more', 'sent_less']].values.tolist()
    
    # Get labels from CSV - required for cal metric 
    if use_actual_labels and 'y' in data.columns:
        labels = data['y'].tolist()
    elif 'stereo_antistereo' in data.columns:
        labels = data['stereo_antistereo'].map({'stereo': 1, 'antistereo': 0}).tolist()
    elif 'label' in data.columns:
        labels = data['label'].tolist()
    else:
        raise ValueError("No label column found in CSV. Need 'y', 'stereo_antistereo', or 'label' column for evaluation.")
    
    return sentence_pairs, labels

def categorize_gender_from_bias_term(bias_term):
    """
    Categorize bias terms into gender groups.
    For gender bias: him/his/he -> male, her/she -> female
    """
    if bias_term is None:
        return None
    
    bias_term_lower = bias_term.lower()
    
    # Gender categorization
    if bias_term_lower in ["him", "his", "he"]:
        return "male"
    elif bias_term_lower in ["her", "she"]:
        return "female"
    else:
        return bias_term

def decide_gender_without_sum(probs1, probs2):
    """
    Categorize sentence pairs into male/female groups for group-ECE calculation.
    
    Compares pronoun probabilities from two sentences to determine gender bias direction.
    Used to split data into male/female groups for measuring calibration bias.
    """
    if len(probs1) == 1 and len(probs2) == 1:
        (p1_pronoun, p1_val) = next(iter(probs1.items()))
        (p2_pronoun, p2_val) = next(iter(probs2.items()))

        if p1_val == p2_val:
            return None  

        if p1_val > p2_val:
            if p1_pronoun in ["him", "his", "he"]:
                return "male"
            elif p1_pronoun in ["her", "she"]:
                return "female"
            return None
        else:
            if p2_pronoun in ["him", "his", "he"]:
                return "male"
            elif p2_pronoun in ["her", "she"]:
                return "female"
            return None
    return None
