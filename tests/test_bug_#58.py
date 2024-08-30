
import os.path

from llama_index.vector_stores.tidbvector import TiDBVectorStore

TIDB_PASSWORD = os.environ.get("TIDB_CLOUD_PASSWORD", "")
TIDB_USER = os.environ.get("TIDB_CLOUD_USER", "root")
TIDB_TABLE = os.environ.get("TIDB_CLOUD_TABLE", "test")

TIDB_DATABASE_URL = f"mysql+pymysql://{TIDB_USER}:{TIDB_PASSWORD}@127.0.0.1:4000/test?ssl_ca=/etc/ssl/cert.pem&ssl_verify_cert=false&ssl_verify_identity=false"

# python -m concave.tools.semantic_search --question "How to create a new table in MySQL?" --top_k 5
if __name__ == "__main__":

    _vector_store = TiDBVectorStore(
        connection_string=TIDB_DATABASE_URL,
        table_name=TIDB_TABLE,
        distance_strategy="cosine",
        vector_dimension=1536,
        drop_existing_table=False,
    )

    raise ValueError("Case triggered for issue#58")
