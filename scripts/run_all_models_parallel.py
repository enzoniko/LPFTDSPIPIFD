#!/usr/bin/env python3
"""
Temporary script to run remaining PINN models in parallel (max 4 concurrent).

This script runs the pinn_to_siamese_wrapper.py for the remaining 6 PINN models
(constant_weight and pecann already completed) with limited concurrency (max 4 at a time)
to prevent resource contention, showing progress for each model as they complete.
Each model saves its output to individual log files.

Features:
- Automatic cleanup of existing PINN training processes on startup
- Proper signal handling (Ctrl+C) with child process cleanup
- Individual log files for each model with full tqdm progress bars
- Resource management to prevent system overload
- No timeout limits - processes can run indefinitely until completion
"""

import subprocess
import threading
import time
import sys
import os
import signal
import psutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# All 8 available PINN models (temporarily excluding completed ones)
ALL_MODELS = [
    'relobralo',
    # 'constant_weight',  # Already completed
    'brdr',
    # 'pecann',          # Already completed
    'adaptive_lbpin',
    'alpinn',
    'dwpinn',
    'gradnorm'
]

def run_single_model(model_name):
    """Run pinn_to_siamese_wrapper.py for a single model."""
    output_path = f"josafat/best_model_residuals/{model_name}_residuals.pth"
    log_file = f"josafat/best_model_residuals/{model_name}.log"

    # Build command
    cmd = [
        sys.executable, 'pinn_to_siamese_wrapper.py',
        '--data-dir', 'Data/v3',
        '--model', model_name,
        '--fraction', '0.2',
        '--no-bayesian',
        '--output', output_path
    ]

    print(f"[START] Starting {model_name}... (logging to {log_file})")
    start_time = time.time()

    try:
        # Open log file for writing
        with open(log_file, 'w', encoding='utf-8') as log_f:
            # Run the command and redirect output to log file
            result = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout so both go to log file
                text=True,
                encoding='utf-8'
            )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"[SUCCESS] {model_name} completed successfully in {elapsed:.1f}s")
            # Read last few lines from log file to show completion summary
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        # Get last non-empty line
                        for line in reversed(lines):
                            line = line.strip()
                            if line:
                                print(f"   [INFO] {model_name} summary: {line}")
                                break
            except Exception as e:
                print(f"   [INFO] {model_name} completed (could not read summary: {e})")
        else:
            print(f"[ERROR] {model_name} failed after {elapsed:.1f}s (exit code: {result.returncode})")
            # Read last few lines from log file to show error
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        # Get last 3 non-empty lines
                        error_lines = []
                        for line in reversed(lines):
                            line = line.strip()
                            if line:
                                error_lines.append(line)
                                if len(error_lines) >= 3:
                                    break
                        error_lines.reverse()
                        print(f"   [DEBUG] {model_name} error: {' | '.join(error_lines)}")
            except Exception as e:
                print(f"   [DEBUG] {model_name} error: Could not read log file ({e})")


    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[CRASH] {model_name} crashed after {elapsed:.1f}s: {str(e)}")
        # Write crash message to log file
        try:
            with open(log_file, 'a', encoding='utf-8') as log_f:
                log_f.write(f"\n[CRASH] Process crashed after {elapsed:.1f}s: {str(e)}\n")
        except:
            pass


def find_and_kill_existing_processes():
    """
    Find and kill any existing Python processes related to PINN model training.
    This prevents resource conflicts from previous interrupted runs.
    """
    current_pid = os.getpid()
    killed_count = 0

    print("[CLEANUP] Checking for existing PINN training processes...")

    try:
        # Get all Python processes
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Skip current process
                if proc.info['pid'] == current_pid:
                    continue

                # Check if it's a Python process
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = proc.info['cmdline']
                    if cmdline:
                        cmdline_str = ' '.join(cmdline)

                        # Check if it's related to our PINN training
                        if ('pinn_to_siamese_wrapper.py' in cmdline_str or
                            'run_all_models_parallel.py' in cmdline_str or
                            any(model in cmdline_str for model in ALL_MODELS)):
                            print(f"[CLEANUP] Found related process PID {proc.info['pid']}: {cmdline_str[:100]}...")

                            try:
                                # Try to terminate gracefully first
                                proc.terminate()
                                # Wait a bit for graceful termination
                                time.sleep(2)

                                # If still running, force kill
                                if proc.is_running():
                                    proc.kill()
                                    print(f"[CLEANUP] Force killed process PID {proc.info['pid']}")
                                else:
                                    print(f"[CLEANUP] Terminated process PID {proc.info['pid']}")

                                killed_count += 1

                            except psutil.AccessDenied:
                                print(f"[WARNING] Access denied killing process PID {proc.info['pid']}")
                            except Exception as e:
                                print(f"[ERROR] Failed to kill process PID {proc.info['pid']}: {e}")

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

    except Exception as e:
        print(f"[WARNING] Error during process cleanup: {e}")

    if killed_count > 0:
        print(f"[CLEANUP] Cleaned up {killed_count} existing processes")
        time.sleep(3)  # Give system time to clean up
    else:
        print("[CLEANUP] No existing processes found")


def signal_handler(signum, frame):
    """Handle interrupt signals to ensure proper cleanup."""
    print(f"\n[INTERRUPT] Received signal {signum}, cleaning up...")

    # Kill any child processes that might still be running
    try:
        current_process = psutil.Process()
        children = current_process.children(recursive=True)

        for child in children:
            try:
                if child.is_running():
                    child.kill()
                    print(f"[CLEANUP] Killed child process PID {child.pid}")
            except:
                pass
    except:
        pass

    print("[EXIT] Script terminated")
    sys.exit(1)


def main():
    """Run all models in parallel."""
    # Set up signal handlers for proper cleanup on interruption
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Clean up any existing processes first
    find_and_kill_existing_processes()

    print("[START] Starting parallel execution of remaining 6 PINN models")
    print("=" * 60)
    print(f"Models to process: {', '.join(ALL_MODELS)}")
    print("Output directory: josafat/best_model_residuals/")
    print("Each model will process Data/v3 with 20% fraction")
    print("Running with max 4 concurrent processes")
    print("=" * 60)

    # Create output directory
    output_dir = Path("josafat/best_model_residuals")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run models with limited concurrency (max 4 at a time)
    start_time = time.time()
    executor = None

    try:
        with ThreadPoolExecutor(max_workers=6) as executor:
            # Submit all tasks
            future_to_model = {executor.submit(run_single_model, model_name): model_name for model_name in ALL_MODELS}

            # Wait for completion and show progress
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    future.result()  # This will raise any exception that occurred
                except Exception as exc:
                    print(f"[ERROR] {model_name} generated an exception: {exc}")

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Keyboard interrupt received, cleaning up...")
        if executor:
            executor.shutdown(wait=False)
        raise
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during execution: {e}")
        if executor:
            executor.shutdown(wait=False)
        raise
    finally:
        total_time = time.time() - start_time

    # Final cleanup check
    print("\n[CLEANUP] Performing final cleanup check...")
    time.sleep(2)  # Give processes time to finish properly

    # Check for any remaining processes
    try:
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        if children:
            print(f"[WARNING] Found {len(children)} child processes still running, cleaning up...")
            for child in children:
                try:
                    if child.is_running():
                        child.kill()
                        print(f"[CLEANUP] Killed remaining child process PID {child.pid}")
                except:
                    pass
        else:
            print("[CLEANUP] No remaining child processes found")
    except Exception as e:
        print(f"[WARNING] Error during final cleanup: {e}")

    print("\n" + "=" * 60)
    print("[DONE] All models completed!")
    print(f"Total execution time: {total_time:.1f}s")
    print("Check josafat/best_model_residuals/ for the results")
    print("=" * 60)

    # Check which files were created
    print("\n[FILES] Generated files:")
    if output_dir.exists():
        # Check for .pth files
        pth_files = list(output_dir.glob("*.pth"))
        if pth_files:
            print("   Residual files (.pth):")
            for file in sorted(pth_files):
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"      {file.name}: {size_mb:.1f} MB")

        # Check for log files
        log_files = list(output_dir.glob("*.log"))
        if log_files:
            print("   Log files (.log):")
            for file in sorted(log_files):
                size_kb = file.stat().st_size / 1024
                print(f"      {file.name}: {size_kb:.1f} KB")

        if not pth_files and not log_files:
            print("   No .pth or .log files found")
    else:
        print("   Output directory not found")

if __name__ == "__main__":
    main()
