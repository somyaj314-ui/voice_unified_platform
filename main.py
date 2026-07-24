import os
import sys

# Ensure the Voice_platform root is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.server import main

if __name__ == "__main__":
    main()
