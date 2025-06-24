import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Azure OpenAI Configuration
    AZURE_ENDPOINT = "https://dancu-mbanolql-eastus2.cognitiveservices.azure.com/"
    AZURE_DEPLOYMENT = "gpt-4o-2"
    AZURE_API_KEY = os.getenv("subscription_key")
    AZURE_API_VERSION = "2024-12-01-preview"
    
    # Data Configuration
    COST_DATA_FILE = 'cost_data.csv'
    DUMMY_DATA_FILE = 'dummy_data.json'
    
    # AI Configuration
    MAX_TOKENS = 1000
    TEMPERATURE = 0.3
    TOP_P = 1.0
    
    # Validation
    @classmethod
    def validate(cls):
        if not cls.AZURE_API_KEY:
            raise ValueError("Please set the subscription_key environment variable in your .env file")
        return True

# Validate configuration on import
Config.validate() 