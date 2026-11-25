from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "churn_db"
    mongo_collection: str = "prediction_logs"
    model_path: str = "churn_model.pkl"

    # reads values from the .env file
    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
