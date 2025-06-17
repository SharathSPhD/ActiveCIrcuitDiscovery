#\!/usr/bin/env python3
import subprocess
import time
import os

def check_experiment_status():
    try:
        # Check if experiment process is running
        result = subprocess.run(['pgrep', '-f', 'run_enhanced_experiment.py'], 
                              capture_output=True, text=True)
        
        if result.stdout.strip():
            print(f"✅ Experiment running (PID: {result.stdout.strip()})")
            
            # Check log file size and last lines
            if os.path.exists('experiment_output.log'):
                size = os.path.getsize('experiment_output.log')
                print(f"📄 Log file size: {size} bytes")
                
                # Show last few lines
                with open('experiment_output.log', 'r') as f:
                    lines = f.readlines()
                    if lines:
                        print("📝 Last 3 log lines:")
                        for line in lines[-3:]:
                            print(f"   {line.strip()}")
            else:
                print("⚠️  No log file found yet")
        else:
            print("❌ Experiment not running")
            
    except Exception as e:
        print(f"Error checking status: {e}")

if __name__ == "__main__":
    check_experiment_status()
