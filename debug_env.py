"""
Debug script to verify .env file loading.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

print("=" * 80)
print("DEBUGGING .ENV FILE LOADING")
print("=" * 80)

# Check current working directory
print(f"\n[1] Current Working Directory: {os.getcwd()}")

# Check if .env exists in various locations
possible_paths = [
    Path.cwd() / ".env",
    Path.cwd() / ".." / ".env",
    Path("/Users/frank/Desktop/Towards AGI/llm-tool-hub/.env"),
]

print(f"\n[2] Checking .env file locations:")
for path in possible_paths:
    exists = path.exists()
    print(f"  - {path}: {'✓ EXISTS' if exists else '✗ NOT FOUND'}")

# Try to load .env manually
env_path = Path.cwd() / ".env"
print(f"\n[3] Attempting to load .env from: {env_path}")
if env_path.exists():
    load_dotenv(env_path)
    print(f"  ✓ Successfully loaded from {env_path}")
else:
    print(f"  ✗ File not found at {env_path}")

# Check environment variables
print(f"\n[4] Checking environment variables:")
env_vars = [
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_MODEL",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
]

for var in env_vars:
    value = os.getenv(var)
    if value:
        if "KEY" in var or "API" in var:
            print(f"  - {var}: {'***' + value[-10:] if len(value) > 10 else '***'}")
        else:
            print(f"  - {var}: {value}")
    else:
        print(f"  - {var}: ✗ NOT SET")

# Print .env file content (masked)
print(f"\n[5] Content of .env file:")
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                if "KEY" in key or "API" in key:
                    print(f"  {key}={'***' + value[-10:] if len(value) > 10 else '***'}")
                else:
                    print(f"  {key}={value}")
