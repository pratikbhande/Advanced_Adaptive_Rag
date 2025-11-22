# config.py
"""Enhanced configuration for Neural-RL Adaptive RAG"""

# Text Processing
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Vector Store
VECTOR_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"

# LLM Models
CHAT_MODEL = "gpt-4o-mini"
ANALYZER_MODEL = "gpt-4o-mini"
CLUSTERING_MODEL = "gpt-4o-mini"
CHAT_TEMPERATURE = 0.7
ANALYZER_TEMPERATURE = 0.0

# Reinforcement Learning
RL_DATA_PATH = "./rl_data"
STRATEGIES = ["concise", "detailed", "structured", "example_driven", "analytical"]
STRATEGY_K_VALUES = {
    "concise": 3,
    "detailed": 5,
    "structured": 4,
    "example_driven": 4,
    "analytical": 6
}

# Neural Network Configuration
NEURAL_INPUT_DIM = 784  # 384 (query) + 384 (cluster) + 1 (complexity) + 10 (user) + 5 (time)
NEURAL_HIDDEN_DIMS = [256, 128, 64]
NEURAL_OUTPUT_DIM = len(STRATEGIES)
NEURAL_DROPOUT = 0.2
NEURAL_LEARNING_RATE = 0.001
BATCH_SIZE = 10
TRAINING_BUFFER_SIZE = 500

# Sentence Transformer
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# RL Parameters
EPSILON_START = 0.35
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.95
UCB_BETA = 2.0
CLUSTER_STRATEGY_BIAS = 0.7

# Concept Drift Detection
DRIFT_WINDOW_SIZE = 20
DRIFT_REFERENCE_SIZE = 30
DRIFT_Z_THRESHOLD = 2.5

# Negative Feedback Handling
CONSECUTIVE_NEGATIVES_THRESHOLD = 3
SUPPRESSION_DURATION = 10
NEGATIVE_REWARD_THRESHOLD = -0.3

# Feedback
POSITIVE_REWARD = 1.0
NEGATIVE_REWARD = -1.0
RECENCY_DECAY = 0.95

# Memory
EPISODIC_MEMORY_SIZE = 50
META_LEARNING_UPDATE_FREQUENCY = 50

# Visualization
PLOT_COLORS = {
    "concise": "#FF6B6B",
    "detailed": "#4ECDC4",
    "structured": "#45B7D1",
    "example_driven": "#FFA07A",
    "analytical": "#98D8C8"
}