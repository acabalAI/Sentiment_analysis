import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Retrieve API keys and other settings from the environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Optional: check for missing keys and raise an error if necessary
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY. Please add it to the .env file.")


