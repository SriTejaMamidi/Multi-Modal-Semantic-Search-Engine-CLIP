from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_ID:str="openai/clip-vit-base-patch32"
    MAX_LABELS:int=50
    MAX_IMAGE_BYTES:int =10 * 1024 * 1024
    class Config:
        env_file = ".env"

settings = Settings()