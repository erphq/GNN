#!/usr/bin/env python3
import sys
import os
import subprocess

# Get the command line arguments
args = sys.argv[1:]

# Print what we're going to run
print(f"Running: python main.py {' '.join(args)}")

# Run the command and capture output
try:
    result = subprocess.run(['python', 'main.py'] + args, 
                           check=True, 
                           text=True,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    print(result.stdout)
    sys.exit(0)
except subprocess.CalledProcessError as e:
    print(f"Error running main.py: {e}")
    print(e.stdout)
    sys.exit(e.returncode)
