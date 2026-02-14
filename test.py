"""
Test MongoDB Connection
Run this before deploying to verify your MongoDB setup
"""
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "fcrules")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "query_stats")


def test_mongodb_connection():
    print("🔍 Testing MongoDB Connection...")
    print(f"Database: {MONGODB_DATABASE}")
    print(f"Collection: {MONGODB_COLLECTION}")
    print()

    if not MONGODB_URI:
        print("❌ ERROR: MONGODB_URI not found in environment variables")
        print("Please set MONGODB_URI in your .env file")
        return False

    try:
        # Test connection
        print("⏳ Connecting to MongoDB...")
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000
        )

        # Ping the database
        client.admin.command('ping')
        print("✅ Successfully connected to MongoDB!")

        # Get database and collection
        db = client[MONGODB_DATABASE]
        collection = db[MONGODB_COLLECTION]

        # Test write operation
        print("\n⏳ Testing write operation...")
        test_doc = {
            "_id": "test_connection",
            "test": True,
            "message": "Connection test successful"
        }
        collection.insert_one(test_doc)
        print("✅ Write operation successful!")

        # Test read operation
        print("\n⏳ Testing read operation...")
        result = collection.find_one({"_id": "test_connection"})
        if result:
            print("✅ Read operation successful!")
            print(f"Retrieved: {result}")

        # Clean up test document
        print("\n⏳ Cleaning up test document...")
        collection.delete_one({"_id": "test_connection"})
        print("✅ Cleanup successful!")

        # Initialize query counter if doesn't exist
        print("\n⏳ Checking query counter...")
        counter_doc = collection.find_one({"_id": "query_counter"})
        if counter_doc:
            print(f"✅ Query counter exists with count: {counter_doc.get('count', 0)}")
        else:
            print("⚠️  Query counter doesn't exist. Initializing...")
            collection.insert_one({
                "_id": "query_counter",
                "count": 0,
                "last_updated": None
            })
            print("✅ Query counter initialized with count: 0")

        # Close connection
        client.close()

        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("=" * 50)
        print("\nYou're ready to deploy to Vercel!")
        return True

    except ConnectionFailure as e:
        print(f"\n❌ CONNECTION FAILED: {e}")
        print("\nCommon issues:")
        print("1. Check your MongoDB URI is correct")
        print("2. Ensure IP whitelist includes 0.0.0.0/0 (for Vercel)")
        print("3. Verify username/password are correct")
        print("4. Check if cluster is running")
        return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        return False


if __name__ == "__main__":
    print("MongoDB Connection Test")
    print("=" * 50)
    test_mongodb_connection()