import os
import sys

# Ensure the Voice_platform root is in the Python path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

try:
    from dotenv import load_dotenv
    # Load global .env first
    load_dotenv(os.path.join(base_dir, ".env"))
    
    # Determine the use case and load its specific .env to override
    use_case = os.getenv("USE_CASE", "admissions")
    use_case_env = os.path.join(base_dir, "use_cases", use_case, ".env")
    
    if os.path.exists(use_case_env):
        print(f"Loading environment from: {use_case_env}")
        load_dotenv(use_case_env, override=True)
except ImportError:
    pass

from core.server import main

if __name__ == "__main__":
    main()
