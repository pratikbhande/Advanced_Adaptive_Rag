INTELLIGENT_QUERY_ANALYSIS_PROMPT = """You are an expert at analyzing user queries across multiple dimensions to understand their true information need.

Analyze the following query across THREE dimensions:

═══════════════════════════════════════════════════════════
DIMENSION 1: INTENT (What is the user trying to learn?)
═══════════════════════════════════════════════════════════

**definition** - User wants to know WHAT something IS (core meaning, basic identity)
   Examples: "What is machine learning?", "Define RFP", "What does API mean?"
   
**explanation** - User wants to understand HOW or WHY something works (mechanisms, reasons)
   Examples: "How does machine learning work?", "Why is RFP important?", "Explain the process"
   
**procedure** - User wants instructions on HOW TO DO something (actionable steps)
   Examples: "How to implement ML?", "Steps to create an RFP", "Guide to using APIs"
   
**comparison** - User wants to understand DIFFERENCES or SIMILARITIES
   Examples: "ML vs AI", "Compare RFP and RFQ", "Difference between REST and GraphQL"
   
**analysis** - User wants EVALUATION, REASONING, or CRITICAL THINKING
   Examples: "Analyze the impact of ML", "Evaluate RFP effectiveness", "Why does X cause Y?"
   
**factual** - User wants SPECIFIC FACTS, DATA, or HISTORICAL INFORMATION
   Examples: "When was ML invented?", "Who created the RFP process?", "How many types of APIs?"

═══════════════════════════════════════════════════════════
DIMENSION 2: DEPTH (How much detail does the user need?)
═══════════════════════════════════════════════════════════

**surface** - Quick, brief answer (1-2 sentences, key point only)
   Indicators: "briefly", "in short", "quick", "summarize", casual phrasing
   Examples: "What's RFP?", "Tell me about ML quick"
   
**moderate** - Standard explanation with context (3-5 sentences, balanced)
   Indicators: Standard question format, no depth modifiers
   Examples: "What is RFP?", "Explain machine learning"
   
**comprehensive** - Detailed, thorough coverage (multiple paragraphs, examples, nuance)
   Indicators: "detailed", "comprehensive", "in depth", "elaborate", "everything about"
   Examples: "Can you tell me everything about RFP?", "Detailed explanation of ML"

═══════════════════════════════════════════════════════════
DIMENSION 3: SCOPE (How focused is the question?)
═══════════════════════════════════════════════════════════

**specific** - Narrow focus on particular aspect
   Indicators: Specific entity/concept named, precise question
   Examples: "What is RFP?", "Define gradient descent"
   
**broad** - Wide-ranging, exploratory, or multiple aspects
   Indicators: "tell me about", "overview", "introduction", "general", multiple topics
   Examples: "Tell me about RFP hunting", "Overview of machine learning field"

═══════════════════════════════════════════════════════════
CRITICAL DISTINCTIONS TO MAKE:
═══════════════════════════════════════════════════════════

❌ WRONG: Treating "What is X?" and "Tell me about X" as the same
✅ RIGHT: 
   - "What is RFP?" → definition_moderate_specific (wants the definition)
   - "Tell me about RFP" → explanation_comprehensive_broad (wants full context)
   - "What's RFP?" → definition_surface_specific (casual, quick answer)

❌ WRONG: Same intent for different phrasings
✅ RIGHT:
   - "Define machine learning" → definition_moderate_specific
   - "How does machine learning work?" → explanation_comprehensive_specific
   - "What can you tell me about ML?" → explanation_comprehensive_broad

═══════════════════════════════════════════════════════════
QUERY TO ANALYZE:
═══════════════════════════════════════════════════════════

"{query}"

═══════════════════════════════════════════════════════════
YOUR TASK:
═══════════════════════════════════════════════════════════

Analyze this query carefully and respond in EXACTLY this format (three lines only):

INTENT: <one of: definition, explanation, procedure, comparison, analysis, factual>
DEPTH: <one of: surface, moderate, comprehensive>
SCOPE: <one of: specific, broad>

Example outputs:
INTENT: definition
DEPTH: moderate
SCOPE: specific

OR

INTENT: explanation
DEPTH: comprehensive
SCOPE: broad

**RESPOND NOW:**"""
