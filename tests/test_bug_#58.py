import argparse
import os.path

from llama_index.vector_stores.tidbvector import TiDBVectorStore

TIDB_PASSWORD = os.environ.get("TIDB_CLOUD_PASSWORD", "")
TIDB_USER = os.environ.get("TIDB_CLOUD_USER", "")
TIDB_TABLE = os.environ.get("TIDB_CLOUD_TABLE", "")

TIDB_DATABASE_URL = f"mysql+pymysql://{TIDB_USER}.root:{TIDB_PASSWORD}@gateway01.us-west-2.prod.aws.tidbcloud.com:4000/test?ssl_ca=/etc/ssl/cert.pem&ssl_verify_cert=false&ssl_verify_identity=false"

# python -m concave.tools.semantic_search --question "How to create a new table in MySQL?" --top_k 5
if __name__ == "__main__":
    query = "code relative create Logging"


    _vector_store = TiDBVectorStore(
        connection_string=TIDB_DATABASE_URL,
        table_name=TIDB_TABLE,
        distance_strategy="cosine",
        vector_dimension=1536,
        drop_existing_table=False,
    )
    from llama_index.embeddings.voyageai import VoyageEmbedding

    if "VOYAGE_API_KEY" not in os.environ:
        raise ValueError(
            "VOYAGE_API_KEY environment variable is not set. Please set it to your Voyage API key."
        )

    _embed_model = VoyageEmbedding(
        model_name="voyage-code-2",
        voyage_api_key=os.environ.get("VOYAGE_API_KEY"),
        truncation=True,
        embed_batch_size=128,
    )

    from llama_index.core.vector_stores import VectorStoreQuery

    query_embedding = _embed_model.get_query_embedding(query)
    query_bundle = VectorStoreQuery(
        query_str=query,
        query_embedding=query_embedding,
        similarity_top_k=6
    )
    result = _vector_store.query(query_bundle)
    print(result)