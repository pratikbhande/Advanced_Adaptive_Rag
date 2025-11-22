QUERY_GENERATION_PROMPT = """You are an expert at generating diverse test queries from a given text document.

═══════════════════════════════════════════════════════════
                    YOUR TASK
═══════════════════════════════════════════════════════════

Generate {num_queries} diverse questions that can be answered using the following text.

REQUIREMENTS:
- Mix of simple, moderate, and complex questions
- Cover different aspects of the text
- Include different question types: factual, conceptual, analytical, comparative
- Questions should be natural and realistic
- Each question on a new line
- Number each question (1., 2., 3., etc.)

═══════════════════════════════════════════════════════════
                    SOURCE TEXT
═══════════════════════════════════════════════════════════

{text}

═══════════════════════════════════════════════════════════
                    GENERATED QUESTIONS
═══════════════════════════════════════════════════════════

Generate {num_queries} questions:"""