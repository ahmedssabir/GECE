#!/usr/bin/env python3
import argparse
import os
import re
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import load_sentence_pairs_from_csv
from evaluation import calculate_bias_ece_macroce_and_ice
from plotting import ReliabilityDiagram

# Memory efficient settings (good for big models)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def build_args():
    parser = argparse.ArgumentParser(
        description="Evaluate bias using sentence pairs."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--model_name", type=str, required=True, help="Pretrained model name or local path")
    parser.add_argument("--output_file", type=str, default=None, help="CSV path for per-item results")
    parser.add_argument("--summary_file", type=str, default=None, help="Text path for summary results")
    parser.add_argument("--bin_output_file", type=str, default=None, help="Text path for ECE bin summary")
    parser.add_argument("--num_bins", type=int, default=10, help="Number of bins for calibration metrics")

    # store_true should default True
    parser.add_argument(
        "--use_actual_labels",
        action="store_true",
        default=True,
        help="Use actual labels from CSV if available",
    )

    parser.add_argument(
        "--plot_reliability",
        action="store_true",
        default=True,
        help="Generate reliability diagram plot with 10 bins",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return name.split("/")[-1].replace("-", "_").replace(":", "_")


def load_model_and_tokenizer(model_name: str):
    # Tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Model load kwargs
    kwargs = {
        "low_cpu_mem_usage": True,
        # Critical to avoid torch<2.6 block when weights are in safetensors
        "use_safetensors": True,
    }

    if torch.cuda.is_available():
        print("Using CUDA:", torch.cuda.get_device_name(0))
        print(f"Available GPUs: {torch.cuda.device_count()}")
        print("Using HuggingFace device_map='auto' for (multi-)GPU placement")
        kwargs["device_map"] = "auto"
    else:
        print("CUDA not available, using CPU")

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception as e:
        msg = str(e)
        # if safetensors is missing
        if "safetensors" in msg.lower() or "use_safetensors" in msg.lower():
            raise RuntimeError(
                "Failed to load with safetensors. This usually means the model checkpoint "
                "does not provide .safetensors weights.\n"
                "Fix options:\n"
                "  1) Upgrade PyTorch to >= 2.6 (recommended), OR\n"
                "  2) Use a model checkpoint that provides safetensors.\n"
                f"Original error:\n{e}"
            )
        raise

    model.eval()
    return model, tokenizer


def main():
    args = build_args()

    # Device for evaluation code paths that still expect a torch.device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Filenames / folders
    model_clean = sanitize_name(args.model_name)
    dataset_clean = os.path.basename(args.dataset).replace(".csv", "").replace("-", "_").replace(":", "_")

    results_folder = f"results_{dataset_clean}"
    os.makedirs(results_folder, exist_ok=True)

    if args.output_file is None:
        args.output_file = os.path.join(results_folder, f"results_{dataset_clean}_{model_clean}.csv")
    if args.summary_file is None:
        args.summary_file = os.path.join(results_folder, f"summary_{dataset_clean}_{model_clean}.txt")
    if args.bin_output_file is None:
        args.bin_output_file = os.path.join(results_folder, f"bins_{dataset_clean}_{model_clean}.txt")

    print("Results will be saved to:")
    print(f"  - Detailed results: {args.output_file}")
    print(f"  - Summary: {args.summary_file}")
    print(f"  - ECE bins: {args.bin_output_file}")

    # Load data
    sentence_pairs, labels = load_sentence_pairs_from_csv(
        args.dataset,
        use_actual_labels=args.use_actual_labels,
    )

    # Load model/tokenizer 
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # Evaluate
    metrics = calculate_bias_ece_macroce_and_ice(
        sentence_pairs=sentence_pairs,
        labels=labels,
        tokenizer=tokenizer,
        model=model,
        output_csv=args.output_file,
        summary_file=args.summary_file,
        bin_output_file=args.bin_output_file,
        num_bins=args.num_bins,
        device=device,
    )

    print(f"\nResults saved to {args.output_file}")

    # Print results
    print("\nFinal Results:")
    print(" - Bias ECE:", metrics["normalized"][0])
    print(" - Bias MacroCE:", metrics["normalized"][1])
    print(" - Bias ICE:", metrics["normalized"][2])
    print(" - Brier Score:", metrics["normalized"][3])
    print(" - Accuracy of Matching Labels:", metrics["normalized"][4])

    print(" - Bucket-wise Metrics:")
    print("   Accuracies:", metrics["bucket_metrics"]["accuracies"])
    print("   Confidences:", metrics["bucket_metrics"]["confidences"])
    print("   Sizes:", metrics["bucket_metrics"]["sizes"])

    # gender-aware metrics
    if "gender_metrics" in metrics:
        print("\n--- Gender-Aware Calibration Metrics ---")
        print(f"Group-ECE (Gender-ECE): {metrics['gender_metrics']['group_ece']:.4f}")
        print(f"Calibration Bias Gap: {metrics['gender_metrics']['bias_gap']:.4f}")
        print(f"ECE (Male): {metrics['gender_metrics']['ece_male']:.4f}")
        print(f"ECE (Female): {metrics['gender_metrics']['ece_female']:.4f}")

    # Reliability diagram
    if args.plot_reliability:
        print("\n--- Generating Reliability Diagram ---")
        results_df = pd.DataFrame(metrics["results"])
        normalized_probs = results_df[["Normalized_Score_more", "Normalized_Score_less"]].values
        true_labels = results_df["True_Label"].values

        reliability_plot_path = os.path.join(results_folder, f"reliability_{dataset_clean}_{model_clean}.png")

        reliability_diagram = ReliabilityDiagram()

        title_display = model_clean.replace("_", "-")
        title_display = re.sub(r"(\d+)b", r"\1B", title_display) 

        reliability_diagram.plot(
            output=normalized_probs,
            labels=true_labels,
            n_bins=10,
            title=title_display,
            save_path=reliability_plot_path,
        )
        print(f"Reliability diagram saved to: {reliability_plot_path}")


if __name__ == "__main__":
    main()

