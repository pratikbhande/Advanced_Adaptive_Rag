# prompt_template/clustering_prompt.py
"""Enhanced prompt for LLM-based semantic query clustering"""

CLUSTERING_PROMPT = """You are an advanced semantic query clustering system for a Retrieval-Augmented Generation (RAG) application. Your task is to intelligently categorize user queries into meaningful semantic clusters.

═══════════════════════════════════════════════════════════
                        CONTEXT
═══════════════════════════════════════════════════════════

USER QUERY:
"{query}"

EXISTING CLUSTERS:
{existing_clusters}

═══════════════════════════════════════════════════════════
                    CLUSTERING GUIDELINES
═══════════════════════════════════════════════════════════

**PRIMARY CONSIDERATIONS:**
1. **Semantic Meaning**: Focus on the underlying intent and topic, not just keywords
2. **Granularity**: Create clusters that are specific enough to be useful but general enough to group related queries
3. **Consistency**: Maintain consistency with existing clusters when appropriate
4. **Scalability**: Avoid creating too many narrow clusters

**WHEN TO ASSIGN TO EXISTING CLUSTER:**
- The query shares the same core topic or domain
- The intent and information need are similar
- The query would benefit from similar retrieval strategies

**WHEN TO CREATE NEW CLUSTER:**
- The query represents a distinctly different topic or domain
- The intent is fundamentally different from existing clusters
- No existing cluster adequately represents the query's semantics

═══════════════════════════════════════════════════════════
                    CLUSTER NAMING RULES
═══════════════════════════════════════════════════════════

✅ GOOD cluster names:
- Descriptive: "machine_learning_concepts", "python_debugging"
- Domain-specific: "medical_diagnosis", "financial_analysis"
- Intent-based: "how_to_tutorials", "comparison_queries"
- Lowercase with underscores: "data_visualization_techniques"

❌ AVOID:
- Too generic: "general", "miscellaneous"
- Too specific: "what_is_gradient_descent_in_neural_networks"
- Mixed case or spaces: "Machine Learning", "data analysis"
- Numbers only: "cluster_1"

═══════════════════════════════════════════════════════════
                        EXAMPLES
═══════════════════════════════════════════════════════════

Example 1:
Query: "What is machine learning?"
Existing: ml_fundamentals, python_programming
→ GROUP: ml_fundamentals
Reason: Core ML concept query

Example 2:
Query: "How to implement gradient descent in Python?"
Existing: ml_fundamentals, python_programming
→ GROUP: ml_algorithm_implementation
Reason: Combines ML algorithm with coding, needs new cluster

Example 3:
Query: "Explain photosynthesis process"
Existing: ml_fundamentals, python_programming
→ GROUP: biology_processes
Reason: Different domain entirely

Example 4:
Query: "Deep learning vs machine learning"
Existing: ml_fundamentals, comparison_queries
→ GROUP: comparison_queries
Reason: Fits existing comparison pattern

Example 5:
Query: "Best practices for data cleaning"
Existing: ml_fundamentals, data_preprocessing
→ GROUP: data_preprocessing
Reason: Matches existing data preparation cluster

═══════════════════════════════════════════════════════════
                    YOUR TASK
═══════════════════════════════════════════════════════════

Analyze the user query and assign it to the most appropriate cluster.

**OUTPUT FORMAT:**
Respond with EXACTLY this format (one line only):

GROUP: cluster_name

**IMPORTANT:**
- Provide ONLY the GROUP line
- No explanations or additional text
- Cluster name must be lowercase with underscores
- Choose the most semantically appropriate cluster

Your response:"""

# Simplified version for faster processing
CLUSTERING_PROMPT_FAST = """Assign this query to an appropriate semantic cluster.

Query: "{query}"

Existing clusters:
{existing_clusters}

Rules:
- Use existing cluster if query is semantically similar
- Create new descriptive cluster if query is distinctly different
- Cluster names: lowercase_with_underscores

Respond with ONLY:
GROUP: cluster_name

Your response:"""