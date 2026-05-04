"""Experiment 9: Figure 10 (two-approach pipeline diagram)."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
