"""Experiment 9: Generate Figure 10 of the article.

Pipeline diagram showing two approaches for computing implied volatility:
1. COS-Brent approach (traditional)
2. Two-ANNs approach (Heston-ANN + IV-ANN)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import plot_pipeline_diagram


def main():
    print("=" * 60)
    print("EXPERIMENT 9: Pipeline Diagram (Figure 10)")
    print("=" * 60)

    print("\n  Generating Figure 10...")
    plot_pipeline_diagram(filename="fig10_two_approaches.png")

    print("\n" + "=" * 60)
    print("EXPERIMENT 9 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
