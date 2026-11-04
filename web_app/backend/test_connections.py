
import os
import boto3

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from botocore.exceptions import ClientError

# Load environment variables from .env
load_dotenv()

# -----------------------------
# 1. Test Supabase PostgreSQL
# -----------------------------

print("\nTesting Supabase PostgreSQL...")

try:
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL is missing from .env")

    # Use the psycopg 3 SQLAlchemy driver
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace(
            "postgresql://",
            "postgresql+psycopg://",
            1
        )

    engine = create_engine(
        database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 15}
    )

    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        print("PostgreSQL connection successful!")
        print("Test query result:", result.scalar())

    engine.dispose()

except Exception as error:
    print("PostgreSQL connection failed!")
    print(type(error).__name__, str(error))


# -----------------------------
# 2. Test Cloudflare R2
# -----------------------------

print("\nTesting Cloudflare R2...")

try:
    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    bucket_name = os.getenv("R2_BUCKET_NAME")
    endpoint_url = os.getenv("R2_ENDPOINT_URL")

    required_values = [
        account_id,
        access_key,
        secret_key,
        bucket_name,
        endpoint_url
    ]

    if not all(required_values):
        raise ValueError("One or more R2 settings are missing from .env")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto"
    )

    s3.head_bucket(Bucket=bucket_name)

    print("Cloudflare R2 connection successful!")
    print("Bucket accessible:", bucket_name)

except Exception as error:
    print("Cloudflare R2 connection failed!")
    print(type(error).__name__, str(error))

print("\nConnection tests completed.")
