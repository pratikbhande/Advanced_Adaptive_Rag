# config.py
"""Production configuration for Neural-RL Adaptive RAG"""

import os
from pathlib import Path

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Paths
BASE_DIR = Path("./data")
VECTOR_DB_PATH = BASE_DIR / "chroma_db"
RL_DATA_PATH = BASE_DIR / "rl_data"
LOGS_PATH = BASE_DIR / "logs"
MODELS_PATH = BASE_DIR / "models"

# Create directories
for path in [BASE_DIR, VECTOR_DB_PATH, RL_DATA_PATH, LOGS_PATH, MODELS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Text Processing
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Embedding
EMBEDDING_MODEL = "text-embedding-3-small"
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# LLM Models
CHAT_MODEL = "gpt-4o-mini"
ANALYZER_MODEL = "gpt-4o-mini"
CHAT_TEMPERATURE = 0.7
ANALYZER_TEMPERATURE = 0.0

# Multi-Dimensional Query Classification
# Intent Types
INTENT_TYPES = [
    "definition",      # Seeking what something IS
    "explanation",     # Seeking HOW/WHY something works
    "procedure",       # Seeking HOW TO do something
    "comparison",      # Seeking differences/similarities
    "analysis",        # Seeking deep reasoning/evaluation
    "factual"         # Seeking specific facts/data
]

# Information Depth
DEPTH_LEVELS = [
    "surface",         # Quick answer, basic info
    "moderate",        # Standard explanation with context
    "comprehensive"    # Detailed, thorough coverage
]

# Query Scope
SCOPE_TYPES = [
    "specific",        # Narrow, focused question
    "broad"           # Wide-ranging, exploratory
]

# Response Strategies
STRATEGIES = ["concise", "detailed", "structured", "example_driven", "analytical"]
STRATEGY_K_VALUES = {
    "concise": 3,
    "detailed": 5,
    "structured": 4,
    "example_driven": 4,
    "analytical": 6
}

# Neural Network Configuration
# Features: query_emb(384) + intent_onehot(6) + depth_onehot(3) + scope_onehot(2) + user_features(15) + time_features(5)
NEURAL_INPUT_DIM = 384 + 6 + 3 + 2 + 10 + 5  # = 410 (FIXED)
NEURAL_HIDDEN_DIMS = [128, 64]
NEURAL_OUTPUT_DIM = len(STRATEGIES)
NEURAL_DROPOUT = 0.3
NEURAL_LEARNING_RATE = 0.001
NEURAL_GAMMA = 0.95
BATCH_SIZE = 16
TRAINING_BUFFER_SIZE = 200
TARGET_UPDATE_FREQUENCY = 50

# RL Parameters
EPSILON_START = 0.3
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
TEMPERATURE_SAMPLING = 1.5

# Experience Replay
REPLAY_ALPHA = 0.6
REPLAY_BETA_START = 0.4
REPLAY_BETA_FRAMES = 1000

# Feedback
POSITIVE_REWARD = 1.0
NEGATIVE_REWARD = -1.0
NEUTRAL_REWARD = 0.0

# Concept Drift Detection
DRIFT_WINDOW_SIZE = 20
DRIFT_REFERENCE_SIZE = 30
DRIFT_THRESHOLD = 0.15

# Strategy Suppression
CONSECUTIVE_NEGATIVES_THRESHOLD = 3
SUPPRESSION_DURATION = 10

# Visualization
PLOT_COLORS = {
    "concise": "#FF6B6B",
    "detailed": "#4ECDC4",
    "structured": "#45B7D1",
    "example_driven": "#FFA07A",
    "analytical": "#98D8C8"
}

INTENT_COLORS = {
    "definition": "#E74C3C",
    "explanation": "#3498DB",
    "procedure": "#2ECC71",
    "comparison": "#F39C12",
    "analysis": "#9B59B6",
    "factual": "#1ABC9C"
}

# Testing Configuration
TEST_QUERIES_PER_TYPE = 5
AUTO_TEST_DELAY = 2.0