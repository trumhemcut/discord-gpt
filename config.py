from pydantic import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    DISCORD_TOKEN: str
    ZAPIER_NLA_API_KEY: str
    DISCORD_TOKEN: str
    RELOAD_DOCUMENTS: bool

    # For Azure Open AI
    # OPENAI_API_TYPE: str
    # DEPLOYMENT_NAME: str
    # MODEL_NAME: str
    
    class Config:
        env_file = ".env"

settings = Settings()
