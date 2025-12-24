#!/usr/bin/env python3
"""
Iterative greedy search agent for StreamingLLM layer selection.

This script:
1. Runs greedy_streamingllm.py to evaluate all layers
2. Identifies the layer with the highest KL divergence
3. Adds that layer to the StreamingLLM configuration
4. Repeats until stopping criteria is met
"""

import subprocess
import sys
import csv
import yaml
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import argparse


@dataclass
class SearchConfig:
    """Configuration for the iterative search process."""
    max_iterations: int = 10
    kl_threshold: Optional[float] = None  # Stop if best KL is above this threshold
    output_dir: Path = Path("search_results")
    base_config_args: List[str] = None  # Arguments to pass to greedy_streamingllm.py


class IterativeSearchAgent:
    """Agent that orchestrates the iterative layer selection process."""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.config.output_dir.mkdir(exist_ok=True, parents=True)
        self.selected_layers: List[int] = []
        self.iteration_history: List[dict] = []
        self.num_layers: Optional[int] = None  # Detected from first iteration

        # Load initial baseline from config files if provided
        self._load_initial_baseline()

    def _load_initial_baseline(self):
        """Load initial layer_hybrid_types from config files if provided."""
        if not self.config.base_config_args:
            return

        # Look for -c or --config arguments
        config_files = []
        i = 0
        while i < len(self.config.base_config_args):
            arg = self.config.base_config_args[i]
            if arg in ['-c', '--config']:
                if i + 1 < len(self.config.base_config_args):
                    config_files.append(self.config.base_config_args[i + 1])
                    i += 2
                else:
                    i += 1
            else:
                i += 1

        # Load and merge config files
        merged_config = {}
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
                    merged_config.update(config_data)
            except Exception as e:
                print(f"Warning: Could not load config file {config_file}: {e}")

        # Extract initial layer_hybrid_types if present
        if 'layer_hybrid_types' in merged_config:
            layer_types = merged_config['layer_hybrid_types']
            if layer_types:
                # Convert to list of layer indices that use StreamingLLM
                self.selected_layers = [
                    i for i, val in enumerate(layer_types) if val
                ]
                if self.selected_layers:
                    print(f"\n[BASELINE] Loaded initial StreamingLLM layers from config: {sorted(self.selected_layers)}\n")

    def run_evaluation(self, iteration: int) -> Tuple[str, int]:
        """
        Run greedy_streamingllm.py and capture output.

        Returns:
            (output_text, return_code)
        """
        # Build command
        cmd = [sys.executable, "greedy_streamingllm.py"]

        # Add base config args if provided
        if self.config.base_config_args:
            cmd.extend(self.config.base_config_args)

        # Create a temporary config file for layer_hybrid_types if needed
        temp_config_file = None
        if self.selected_layers:
            # Use detected num_layers, or infer from selected layers
            if self.num_layers is None:
                # If we don't know yet, use a conservative estimate
                max_layer = max(self.selected_layers) if self.selected_layers else 0
                num_layers = max_layer + 1
            else:
                num_layers = self.num_layers

            layer_hybrid_types = [
                1 if i in self.selected_layers else 0
                for i in range(num_layers)
            ]

            # Create temporary YAML config with layer_hybrid_types
            temp_config = {"layer_hybrid_types": layer_hybrid_types}
            temp_config_file = self.config.output_dir / f"_temp_iter{iteration}.yaml"

            with open(temp_config_file, 'w') as f:
                yaml.dump(temp_config, f)

            # Add the temp config file to the command (it will override other configs)
            cmd.extend(["-c", str(temp_config_file)])

        print(f"\n{'='*80}")
        print(f"Iteration {iteration}: Running evaluation...")
        if self.selected_layers:
            print(f"Current StreamingLLM layers ({len(self.selected_layers)} total): {sorted(self.selected_layers)}")
        else:
            print("Current StreamingLLM layers: None (baseline full attention)")
        print(f"{'='*80}\n")

        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        # Save output to file
        output_file = self.config.output_dir / f"iter{iteration}.txt"
        with open(output_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"\n{'='*80}\nSTDOUT:\n{'='*80}\n")
            f.write(result.stdout)
            f.write(f"\n{'='*80}\nSTDERR:\n{'='*80}\n")
            f.write(result.stderr)

        print(f"Output saved to {output_file}")

        return result.stdout + result.stderr, result.returncode

    def parse_results(self, output: str, iteration: int) -> List[Tuple[int, float]]:
        """
        Parse the CSV output from greedy_streamingllm.py.

        Returns:
            List of (layer_id, kl_divergence) tuples
        """
        results = []

        # Find lines that look like CSV: "layer_id,kl_div_loss"
        for line in output.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Skip header line
            if line == "layer_id,kl_div_loss":
                continue

            # Try to parse as CSV
            parts = line.split(',')
            if len(parts) == 2:
                try:
                    layer_id = int(parts[0])
                    kl_div = float(parts[1])
                    results.append((layer_id, kl_div))
                except (ValueError, IndexError):
                    continue

        # Save parsed results to CSV
        csv_file = self.config.output_dir / f"iter{iteration}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['layer_id', 'kl_div_loss'])
            writer.writerows(results)

        print(f"Parsed {len(results)} layer results")
        print(f"Results saved to {csv_file}")

        # Update num_layers from first iteration
        if self.num_layers is None and results:
            self.num_layers = max(layer_id for layer_id, _ in results) + 1
            print(f"Detected {self.num_layers} layers in model")

        return results

    def find_best_layer(self, results: List[Tuple[int, float]]) -> Optional[Tuple[int, float]]:
        """
        Find the layer with the highest KL divergence that's not already selected.

        Returns:
            (layer_id, kl_divergence) or None if no valid results
        """
        if not results:
            return None

        # Sort by KL divergence (descending - highest first)
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

        # Find the best layer that's NOT already in selected_layers
        for layer_id, kl_div in sorted_results:
            if layer_id not in self.selected_layers:
                return (layer_id, kl_div)

        # All layers are already selected
        return None

    def should_stop(self, iteration: int, best_kl: float) -> Tuple[bool, str]:
        """
        Determine if the search should stop.

        Returns:
            (should_stop, reason)
        """
        # Check max iterations
        if iteration >= self.config.max_iterations:
            return True, f"Reached max iterations ({self.config.max_iterations})"

        # Check KL threshold
        if self.config.kl_threshold is not None and best_kl > self.config.kl_threshold:
            return True, f"Best KL divergence ({best_kl:.6f}) exceeds threshold ({self.config.kl_threshold})"

        return False, ""

    def run(self) -> List[int]:
        """
        Run the iterative search process.

        Returns:
            List of selected layer IDs
        """
        print(f"\n{'#'*80}")
        print(f"# Starting Iterative StreamingLLM Layer Search")
        print(f"# Max iterations: {self.config.max_iterations}")
        if self.config.kl_threshold:
            print(f"# KL threshold: {self.config.kl_threshold}")
        print(f"# Output directory: {self.config.output_dir}")
        print(f"{'#'*80}\n")

        for iteration in range(1, self.config.max_iterations + 1):
            # Run evaluation
            output, return_code = self.run_evaluation(iteration)

            if return_code != 0:
                print(f"\n[ERROR] Evaluation failed with return code {return_code}")
                print("Check the output file for details.")
                break

            # Parse results
            results = self.parse_results(output, iteration)

            if not results:
                print(f"\n[ERROR] No valid results found in output")
                break

            # Find best layer (that's not already selected)
            best_result = self.find_best_layer(results)

            if best_result is None:
                print(f"\n[STOPPING] All layers are already using StreamingLLM")
                break

            best_layer, best_kl = best_result

            print(f"\n{'='*80}")
            print(f"Iteration {iteration} Results:")
            print(f"  Best NEW layer to add: {best_layer}")
            print(f"  KL divergence: {best_kl:.6f}")
            print(f"{'='*80}\n")

            # Save iteration history
            self.iteration_history.append({
                'iteration': iteration,
                'best_layer': best_layer,
                'best_kl': best_kl,
                'selected_layers_before': self.selected_layers.copy(),
            })

            # Check stopping criteria
            should_stop, reason = self.should_stop(iteration, best_kl)
            if should_stop:
                print(f"\n[STOPPING] {reason}")
                break

            # Add best layer to selected layers
            self.selected_layers.append(best_layer)
            print(f"[ACTION] Adding layer {best_layer} to StreamingLLM layers")
            print(f"[STATUS] Updated StreamingLLM layers ({len(self.selected_layers)} total): {sorted(self.selected_layers)}\n")

        # Save final summary
        self.save_summary()

        return sorted(self.selected_layers)

    def save_summary(self):
        """Save a summary of the search process."""
        summary_file = self.config.output_dir / "summary.txt"

        with open(summary_file, 'w') as f:
            f.write("Iterative StreamingLLM Layer Search Summary\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total iterations: {len(self.iteration_history)}\n")
            f.write(f"Final StreamingLLM layers: {sorted(self.selected_layers)}\n\n")

            # Layer importance ranking
            f.write("Layer Importance (Selection Order):\n")
            f.write("-" * 80 + "\n")
            for idx, hist in enumerate(self.iteration_history, 1):
                f.write(f"  {idx}. Layer {hist['best_layer']:2d}  (KL divergence: {hist['best_kl']:.10f})\n")
            f.write("\n")

            f.write("Iteration History:\n")
            f.write("-" * 80 + "\n")

            for hist in self.iteration_history:
                f.write(f"\nIteration {hist['iteration']}:\n")
                f.write(f"  Layers before: {sorted(hist['selected_layers_before'])}\n")
                f.write(f"  Best layer found: {hist['best_layer']}\n")
                f.write(f"  KL divergence: {hist['best_kl']:.10f}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Final configuration: {sorted(self.selected_layers)}\n")

        print(f"\nSearch summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Iterative greedy search for StreamingLLM layer selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python iterative_search.py

  # Specify max iterations and pass args to greedy_streamingllm.py
  python iterative_search.py --max_iterations 5 -- --num_gpus 4 --max_iters 100

  # With KL threshold
  python iterative_search.py --kl_threshold 0.001 -- --num_gpus 2
        """
    )

    parser.add_argument(
        '--max_iterations',
        type=int,
        default=10,
        help='Maximum number of iterations (default: 10)'
    )

    parser.add_argument(
        '--kl_threshold',
        type=float,
        default=None,
        help='Stop if best KL divergence exceeds this threshold (default: None)'
    )

    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('search_results'),
        help='Directory to save results (default: search_results)'
    )

    # Remaining arguments are passed to greedy_streamingllm.py
    parser.add_argument(
        'base_config_args',
        nargs='*',
        help='Arguments to pass to greedy_streamingllm.py (use -- to separate)'
    )

    args = parser.parse_args()

    # Create config
    config = SearchConfig(
        max_iterations=args.max_iterations,
        kl_threshold=args.kl_threshold,
        output_dir=args.output_dir,
        base_config_args=args.base_config_args if args.base_config_args else []
    )

    # Create and run agent
    agent = IterativeSearchAgent(config)
    final_layers = agent.run()

    print(f"\n{'#'*80}")
    print(f"# Search Complete!")
    print(f"# Final StreamingLLM layers: {final_layers}")
    print(f"# Results saved to: {config.output_dir}")
    print(f"{'#'*80}\n")


if __name__ == '__main__':
    main()
