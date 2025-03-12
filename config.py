import os
import logging



# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("⚠️ Missing API Key! Set GROQ_API_KEY in .env or environment variables.")

# Model Configurations
DRAFT_MODEL = "llama3-70b-8192"
MAIN_MODEL = "llama3-8b-8192"

# Setup Logging
def setup_logging():
    """Initialize and return a logger instance."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)

# Ensure logging is set up when the module is imported
logger = setup_logging()
