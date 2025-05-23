"""Default configurations for AutoQED"""

# AutoD defaults
from typing import Any

INPUT_TYPE = "csv"
FILE_ENCODING = "utf-8"
TEXT_COLUMN = "text"
ENCODING_MODEL = "o200k_base"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
EMBEDDING_BATCH_SIZE = 32
NUM_CLUSTERS = 100
NUM_SAMPLES_PER_CLUSTER = 10
RANDOM_SEED = 42

DOCUMENT_OUTPUT = "documents"
TEXT_UNIT_OUTPUT = "text_units"
SAMPLE_TEXTS_OUTPUT = "sample_texts"

# General LLM defaults
LLM_PARAMS: dict[str, Any] = {"temperature": 0.0, "seed": 42}
MAX_DATA_TOKENS = 8000

# AutoQ defaults
NUM_QUESTIONS = 50
OVERSAMPLE_FACTOR = 2.0
