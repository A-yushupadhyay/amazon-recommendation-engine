"""
dynamo_writer.py

Phase 2: Takes the item_similarity.parquet produced by similarity_engine.py
and writes it into DynamoDB Local as the online serving store.

Table design: partition key = item_id, single attribute holding the full
list of {similar_item_id, score} for that item. This is a single-key
lookup pattern -- exactly DynamoDB's sweet spot, and mirrors how a real
"customers who bought this also bought" widget would be served.

Usage:
    python ml/dynamo_writer.py --input data/processed/item_similarity.parquet
"""

import argparse
import os
import boto3
import pandas as pd
from decimal import Decimal
from dotenv import load_dotenv

load_dotenv()

TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "ItemSimilarity")
ENDPOINT = os.getenv("DYNAMODB_ENDPOINT", "http://localhost:8000")
REGION = os.getenv("AWS_REGION", "us-east-1")


def get_dynamo_resource():
    return boto3.resource(
        "dynamodb",
        endpoint_url=ENDPOINT,
        region_name=REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "fakeMyKeyId"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "fakeSecretAccessKey"),
    )


def create_table_if_not_exists(dynamodb):
    existing_tables = [t.name for t in dynamodb.tables.all()]
    if TABLE_NAME in existing_tables:
        print(f"Table '{TABLE_NAME}' already exists, skipping creation.")
        return dynamodb.Table(TABLE_NAME)

    print(f"Creating table '{TABLE_NAME}'...")
    table = dynamodb.create_table(
        TableName=TABLE_NAME,
        KeySchema=[{"AttributeName": "item_id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "item_id", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    table.wait_until_exists()
    print("Table created.")
    return table


def write_similarity_data(table, similarity_df: pd.DataFrame):
    """
    Group the flat similarity_df (item_id, similar_item_id, score) by
    item_id, then batch-write one item per DynamoDB item: {item_id, similar_items: [...]}
    """
    grouped = similarity_df.groupby("item_id")

    print(f"Writing {grouped.ngroups} items to DynamoDB (batched)...")
    with table.batch_writer() as batch:
        for i, (item_id, group) in enumerate(grouped):
            similar_items = [
                {"item_id": row.similar_item_id, "score": Decimal(str(round(row.score, 6)))}
                for row in group.itertuples()
            ]
            batch.put_item(Item={"item_id": item_id, "similar_items": similar_items})
            if (i + 1) % 500 == 0:
                print(f"  ...{i + 1}/{grouped.ngroups} items written")

    print("Done writing to DynamoDB.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to item_similarity.parquet")
    args = parser.parse_args()

    print(f"Loading similarity data from {args.input}...")
    similarity_df = pd.read_parquet(args.input)
    print(f"Loaded {len(similarity_df)} similarity rows")

    dynamodb = get_dynamo_resource()
    table = create_table_if_not_exists(dynamodb)
    write_similarity_data(table, similarity_df)

    sample_item_id = similarity_df["item_id"].iloc[0]
    response = table.get_item(Key={"item_id": sample_item_id})
    if "Item" in response:
        print(f"\nSanity check passed. Sample item '{sample_item_id}' has "
              f"{len(response['Item']['similar_items'])} similar items in DynamoDB.")
    else:
        print("\nSanity check FAILED — item not found after write.")


if __name__ == "__main__":
    main()
