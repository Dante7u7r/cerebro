#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IA VIVA - MASTER EXPERIMENT RUNNER
# Runs all 19 experiments sequentially in separate processes to prevent Windows DLL lock issues.
"""

import os
import sys
import time
import subprocess

def main():
    start_time = time.time()
    experiments = [19]
    timings = {}
    failures = []
    
    print("=" * 70)
    print("  IA VIVA - SEQUENTIAL EXPERIMENT RUNNER (1-19)")
    print("=" * 70)
    print(f"Using interpreter: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    print("-" * 70)
    
    for exp_num in experiments:
        exp_start = time.time()
        print(f"\n>>> Starting Experiment {exp_num}/{len(experiments)}...")
        
        try:
            # Run the experiment in an unbuffered subprocess to get real-time prints
            result = subprocess.run(
                [sys.executable, "-u", "cerebro_brian2.py", str(exp_num)],
                check=True
            )
            elapsed = time.time() - exp_start
            timings[exp_num] = elapsed
            print(f">>> Experiment {exp_num} completed successfully in {elapsed:.2f} seconds.")
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - exp_start
            timings[exp_num] = elapsed
            failures.append(exp_num)
            print(f"!!! Experiment {exp_num} FAILED after {elapsed:.2f} seconds with exit code {e.returncode}.")
            # Ask if we should continue, or just keep going
            print("Continuing with next experiments...")
            
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("  SIMULATION RUN SUMMARY")
    print("=" * 70)
    print(f"{'Exp #':<7s} | {'Status':<10s} | {'Time (s)':<10s}")
    print("-" * 70)
    for exp_num in experiments:
        status = "FAILED" if exp_num in failures else "SUCCESS"
        elapsed_str = f"{timings.get(exp_num, 0.0):.2f}"
        print(f"Exp {exp_num:<3d} | {status:<10s} | {elapsed_str:<10s}")
    print("-" * 70)
    print(f"Total Suite Execution Time: {total_time/60.0:.2f} minutes ({total_time:.2f} seconds)")
    
    if failures:
        print(f"Failed experiments: {failures}")
        sys.exit(1)
    else:
        print("All experiments completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
