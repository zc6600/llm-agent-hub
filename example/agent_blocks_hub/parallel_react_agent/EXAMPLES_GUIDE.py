"""
Quick reference guide for Parallel React Agent examples.

This file documents the 4 concrete examples and how to use them.
"""

# ===== EXAMPLES OVERVIEW =====

"""
We provide 4 progressive examples from basic to real-world applications:

1. 01_basic_usage.py
   - Basic 3-query example
   - Shows fundamental usage pattern
   - ~5 minutes runtime

2. 02_ai_safety_research.py
   - Multi-angle AI safety research (Academic/Research)
   - 3 research angles in parallel
   - Comprehensive output with statistics
   - ~10-15 minutes runtime

3. 03_competitive_analysis.py
   - Competitive intelligence analysis (Business)
   - Market analysis from 3 angles
   - Synthesis into actionable insights
   - ~10-15 minutes runtime

4. 04_product_evaluation.py
   - Technology evaluation framework (Decision Support)
   - Evaluate products across multiple dimensions
   - Cost-benefit analysis included
   - ~10-15 minutes runtime
"""

# ===== USAGE PATTERNS =====

"""
Each example follows this pattern:

1. INITIALIZATION
   - Create LLM instance
   - Initialize search tools
   
2. DESIGN
   - Define research/analysis queries
   - Structure investigation angles
   
3. CONFIGURATION
   - Create system prompt with guidelines
   - Define evaluation criteria
   
4. AGENT CREATION
   - Create parallel_react_agent with LLM + tools + prompt
   
5. EXECUTION
   - Call agent.invoke() with queries
   
6. RESULTS DISPLAY
   - Show individual agent results
   - Display synthesized summary
   
7. ANALYSIS
   - Interpret findings
   - Extract actionable insights
"""

# ===== WHEN TO USE EACH EXAMPLE =====

"""
01_basic_usage.py
├─ Purpose: Get started quickly
├─ Audience: New users, testing setup
├─ Use when: Learning the basic pattern
└─ Expected time: 5 minutes

02_ai_safety_research.py
├─ Purpose: Research complex topics from multiple angles
├─ Audience: Researchers, analysts, knowledge workers
├─ Use when: 
│  ├─ Need comprehensive topic overview
│  ├─ Want to avoid single-perspective bias
│  └─ Researching multi-faceted issues
├─ Industry: Academia, Research, Consulting
└─ Expected time: 10-15 minutes

03_competitive_analysis.py
├─ Purpose: Analyze market and competitors
├─ Audience: Product managers, business analysts
├─ Use when:
│  ├─ Making strategic decisions
│  ├─ Evaluating market positioning
│  └─ Competitive benchmarking needed
├─ Industry: Business, Product, Strategy
└─ Expected time: 10-15 minutes

04_product_evaluation.py
├─ Purpose: Evaluate technology options
├─ Audience: Tech leads, architects, procurement
├─ Use when:
│  ├─ Choosing between technologies
│  ├─ Making procurement decisions
│  └─ ROI and cost analysis needed
├─ Industry: Engineering, IT, Operations
└─ Expected time: 10-15 minutes
"""

# ===== ADAPTING EXAMPLES =====

"""
To adapt these examples for your needs:

1. Change the queries:
   query_1 = "Your first angle here"
   query_2 = "Your second angle here"
   query_3 = "Your third angle here"

2. Update the system prompt:
   custom_instructions = \"\"\"
   Your specific guidelines here
   \"\"\"

3. Change the tools:
   Instead of DuckDuckGoSearchRun, use:
   - ArxivQueryRun for academic papers
   - Custom tools for proprietary data
   - Multiple tools in list: [tool1, tool2]

4. Adjust agent configuration:
   agent = create_parallel_react_agent(
       llm=llm,
       tools=your_tools,
       system_prompt=your_instructions,
   )

5. Process results:
   result = agent.invoke({
       "parallel_react_agent_messages": your_queries
   })
   
   Access results:
   - result["agent_results"]     # Individual findings
   - result["final_summary"]     # Synthesized summary
"""

# ===== EXAMPLE CUSTOMIZATIONS =====

"""
Research Use Case (like 02_ai_safety_research.py):
├─ Queries: Different aspects of topic
├─ Tools: Academic search, paper databases
├─ Prompt: Guidelines for research quality
└─ Output: Comprehensive report

Business Use Case (like 03_competitive_analysis.py):
├─ Queries: Market, pricing, positioning angles
├─ Tools: Web search, financial data
├─ Prompt: Business analysis framework
└─ Output: Competitive intelligence report

Technical Use Case (like 04_product_evaluation.py):
├─ Queries: Technical specs, user feedback, costs
├─ Tools: Tech blogs, review sites, pricing pages
├─ Prompt: Technical evaluation criteria
└─ Output: Decision matrix and recommendation
"""

# ===== PERFORMANCE TIPS =====

"""
To optimize your parallel agent runs:

1. Query Design:
   ✓ Make queries specific and clear
   ✓ Avoid redundant queries
   ✓ Use parallel-friendly queries (independent)
   
2. Tool Selection:
   ✓ Fast tools for quick results
   ✓ Reliable tools for consistent data
   ✓ Consider rate limits and quotas
   
3. Prompt Engineering:
   ✓ Clear, specific instructions
   ✓ Define output format
   ✓ Include evaluation criteria
   
4. LLM Selection:
   ✓ Claude/GPT-4 for quality
   ✓ Consider cost vs quality
   ✓ Test with different models
"""

# ===== EXTENDING WITH MORE QUERIES =====

"""
The system is flexible - you can add more queries:

Instead of 3 queries:
queries = [
    "Query 1",
    "Query 2",
    "Query 3",
]

Use more queries:
queries = [
    "Query 1 - Angle A",
    "Query 2 - Angle B",
    "Query 3 - Angle C",
    "Query 4 - Angle D",
    "Query 5 - Angle E",
]

The system automatically:
- Creates N parallel agents
- Processes them simultaneously
- Synthesizes all results
- Returns comprehensive summary

More angles = more comprehensive analysis
But also = longer processing time
"""

# ===== TROUBLESHOOTING =====

"""
Issue: Queries take too long
Solution: Simplify queries, use faster LLM, adjust max_iterations

Issue: Results not specific enough
Solution: Add more detail to system_prompt, refine query wording

Issue: Synthesis missing key points
Solution: Add specific synthesis instructions to system_prompt

Issue: Tool errors
Solution: Try different tools, add error handling, test tools separately

Issue: API rate limits
Solution: Add delays between queries, use tools with rate limits
"""

if __name__ == "__main__":
    print("📖 Parallel React Agent Examples Guide")
    print("=====================================")
    print()
    print("Quick Start:")
    print("1. Start with 01_basic_usage.py")
    print("2. Try domain-specific examples (02, 03, 04)")
    print("3. Adapt to your use case")
    print()
    print("For detailed examples, see README.md")
