# prompt_template/analysis_prompts.py
"""Prompts for query analysis and understanding"""

# ============================================================================
#                    QUERY COMPLEXITY ANALYSIS
# ============================================================================

QUERY_COMPLEXITY_PROMPT = """You are an expert at analyzing query complexity. Evaluate the following query and determine its complexity level.

═══════════════════════════════════════════════════════════
                    COMPLEXITY CRITERIA
═══════════════════════════════════════════════════════════

**SIMPLE:**
- Single, straightforward concept
- Requires basic factual information
- Can be answered in 1-2 sentences
- No multi-step reasoning needed
- Examples: "What is X?", "Define Y", "When did Z happen?"

**MODERATE:**
- Multiple related concepts
- Requires explanation or description
- May need 3-5 sentences to answer well
- Some conceptual understanding needed
- Examples: "How does X work?", "Explain Y", "What are the benefits of Z?"

**COMPLEX:**
- Multiple interconnected concepts
- Requires analysis, comparison, or synthesis
- Needs detailed explanation (5+ sentences)
- Multi-step reasoning required
- Involves evaluation or critical thinking
- Examples: "Compare X and Y", "Analyze the impact of Z", "Evaluate the relationship between A and B"

═══════════════════════════════════════════════════════════
                        QUERY TO ANALYZE
═══════════════════════════════════════════════════════════

Query: {query}

═══════════════════════════════════════════════════════════
                        YOUR TASK
═══════════════════════════════════════════════════════════

Analyze the query and respond with EXACTLY ONE WORD:
- simple
- moderate
- complex

No explanation needed. Just the single word.

Complexity:"""


# ============================================================================
#                    QUERY INTENT ANALYSIS
# ============================================================================

QUERY_INTENT_PROMPT = """You are an expert at understanding user intent. Analyze the following query and determine the user's primary intent.

═══════════════════════════════════════════════════════════
                        INTENT CATEGORIES
═══════════════════════════════════════════════════════════

**FACTUAL**: Seeking specific factual information
- Examples: "What is X?", "When did Y happen?", "Who created Z?"

**PROCEDURAL**: Seeking instructions or how-to information
- Examples: "How to do X?", "Steps for Y", "Guide to Z"

**CONCEPTUAL**: Seeking understanding of concepts or explanations
- Examples: "How does X work?", "Explain Y", "Why does Z occur?"

**COMPARATIVE**: Seeking comparison or evaluation
- Examples: "X vs Y", "Compare A and B", "What's the difference between C and D?"

**ANALYTICAL**: Seeking analysis or deeper insights
- Examples: "Analyze X", "Evaluate Y", "What are the implications of Z?"

**EXPLORATORY**: Open-ended exploration of a topic
- Examples: "Tell me about X", "Overview of Y", "Introduction to Z"

═══════════════════════════════════════════════════════════
                        QUERY TO ANALYZE
═══════════════════════════════════════════════════════════

Query: {query}

═══════════════════════════════════════════════════════════
                        YOUR TASK
═══════════════════════════════════════════════════════════

Determine the primary intent and respond with EXACTLY ONE WORD:
- factual
- procedural
- conceptual
- comparative
- analytical
- exploratory

No explanation needed. Just the single word.

Intent:"""