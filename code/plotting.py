#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

class ReliabilityDiagram:
    def compute(self, output, labels, n_bins=15):
        if output.shape[1] == 2:
            predictions = np.ones(output.shape[0], dtype="int")
            confidences = output[:, 0]
        else:
            predictions = np.argmax(output, axis=1)
            confidences = np.max(output, axis=1)

        accuracies = (predictions == labels).astype(float)
        bin_indices = (confidences * n_bins).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        bin_confidences = [[] for _ in range(n_bins)]
        bin_accuracies = [[] for _ in range(n_bins)]

        for idx, bin_idx in enumerate(bin_indices):
            bin_confidences[bin_idx].append(confidences[idx])
            bin_accuracies[bin_idx].append(accuracies[idx])

        self.bin_confidences = [np.mean(b) if b else 0 for b in bin_confidences]
        self.bin_accuracies = [np.mean(b) if b else 0 for b in bin_accuracies]

    def plot(self, output, labels, n_bins=15, title="Model", save_path=None):
        self.compute(output, labels, n_bins)

        delta = 1.0 / n_bins
        x = np.arange(0, 1, delta)
        mid = np.linspace(delta / 2, 1 - delta / 2, n_bins)
        error = np.abs(np.subtract(mid, self.bin_accuracies))

        plt.figure(figsize=(7, 7))

        # Accuracy bars
        plt.bar(
            x,
            self.bin_accuracies,
            color='#5B8FA8',
            width=delta,
            align='edge',
            edgecolor='black',
            label='Outputs'
        )

        # Calibration gap
        plt.bar(
            x,
            error,
            bottom=np.minimum(self.bin_accuracies, mid),
            color='#ff9999',
            alpha=0.8,
            width=delta,
            align='edge',
            edgecolor='black',
            hatch='///',
            label='Calibration Gap'
        )

        # Perfect calibration line
        #plt.plot([0, 1], [0, 1], linestyle='--', color='blue', linewidth=2, label='Perfect Calibration')
        plt.plot([0, 1], [0, 1], linestyle='--', color='#4d4d4d', linewidth=2, label='Perfect Calibration')


        plt.xlabel('Confidence', fontsize=26)
        plt.ylabel('Accuracy', fontsize=26)
        plt.title(title, fontsize=24)
        plt.xticks(np.linspace(0, 1, 6), fontsize=22)
        plt.yticks(np.linspace(0, 1, 6), fontsize=22)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(loc='upper left', fontsize=20)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Reliability diagram saved to {save_path}")
        else:
            plt.show()
