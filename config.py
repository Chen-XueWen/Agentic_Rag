import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    TOGETHER_API_KEY: str = os.getenv("TOGETHER_API_KEY")
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    DATASET_NAME: str = "agents-course/unit3-invitees"
    DATASET_SPLIT: str = "train"

settings = Settings()