#\!/usr/bin/env python3
import subprocess
import sys
import os

os.chdir('/home/ubuntu/ActiveCIrcuitDiscovery')

venv_python = '/home/ubuntu/project_venv/bin/python3'
script = 'run_comprehensive_experiment.py'
args = sys.argv[1:]

try:
    cmd = [venv_python, script] + args
    result = subprocess.run(cmd, 
                          env=dict(os.environ, 
                                 PATH='/home/ubuntu/project_venv/bin:' + os.environ.get('PATH', '')),
                          timeout=600,
                          capture_output=False)
    sys.exit(result.returncode)
except subprocess.TimeoutExpired:
    print('Experiment timed out after 10 minutes')
    sys.exit(1)
except Exception as e:
    print('Error running experiment:', e)
    sys.exit(1)
