from motor.motor_asyncio import AsyncIOMotorClient

from app.config import settings

client: AsyncIOMotorClient | None = None


async def connect_to_mongo() -> None:
    """Open the connection once when the app starts."""
    global client
    client = AsyncIOMotorClient(settings.mongo_uri, serverSelectionTimeoutMS=5000, tz_aware=True)
    await client.admin.command("ping")  # fails fast if MongoDB is not reachable
    print("Connected to MongoDB")


async def close_mongo_connection() -> None:
    """Close the connection when the app stops."""
    if client is not None:
        client.close()
        print("MongoDB connection closed")


def get_logs_collection():
    """Return the prediction_logs collection."""
    return client[settings.mongo_db_name][settings.mongo_collection]
