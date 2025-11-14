# Deep Diver Agent Design Document

## Overview

The Deep Diver agent is a LangGraph-based research agent that implements the scientific method for systematic knowledge discovery. It combines Karl Popper's philosophy of science with Lean Startup methodology to create a rigorous, iterative research workflow.

## Philosophy

### Karl Popper's Scientific Method

- **Falsificationism**: Hypotheses should be falsifiable, not just confirmable
- **Critical Rationalism**: Knowledge grows through criticism and testing
- **Problem-Oriented**: Start with clear, well-defined problems

### Lean Startup Methodology

- **Build-Measure-Learn**: Iterative hypothesis testing
- **Validated Learning**: Learn from both successes and failures
- **Pivot or Persevere**: Refine hypotheses based on evidence

## Architecture

### State Management

The agent uses a typed state (`DeepDiverState`) that tracks:

- Original question and decomposed problems
- Gathered information from tools
- Hypotheses with verification status
- Experience pool of validated knowledge
- Iteration count and limits
- Final answer with confidence score

### Workflow Nodes

1. **Formulate Problem** (`formulate_problem`)

   - Input: User question
   - Output: Decomposed sub-problems
   - Purpose: Break complex questions into testable components
2. **Gather Information** (`gather_information`)

   - Input: Decomposed problems
   - Output: Relevant information from tools
   - Purpose: Collect evidence for hypothesis generation
3. **Generate Hypothesis** (`generate_hypothesis`)

   - Input: Gathered information
   - Output: Testable hypotheses
   - Purpose: Create falsifiable explanations
4. **Verify Hypothesis** (`verify_hypothesis`)

   - Input: Hypotheses and evidence
   - Output: Verification results, updated experience pool
   - Purpose: Test and validate hypotheses
5. **Final Answer** (`final_answer`)

   - Input: Verified hypotheses, experience pool
   - Output: Comprehensive answer with confidence
   - Purpose: Synthesize knowledge into answer

### Control Flow

```
Start → Initialize → Formulate Problem → Gather Information → Generate Hypothesis
                                                                      ↓
                                                                Verify Hypothesis
                                                                      ↓
                                                            [LLM Decision Point]
                                                             /              \
                                      CONTINUE (iterate again)              FINISH
                                             ↓                                ↓
                                      Generate Hypothesis              Final Answer → End
```

**Key Innovation**: The decision to continue or finish is made by the LLM itself, not hard-coded rules. The LLM evaluates:

- Quality and confidence of verified hypotheses
- Whether knowledge gaps remain
- If additional iterations would provide meaningful improvement
- Diminishing returns consideration

## API Design

### Main Interface

```python
def create_deepdiver_agent(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: Optional[str] = None,
    max_iterations: int = 3
) -> CompiledGraph
```

### Usage Pattern

```python
# Create agent
agent = create_deepdiver_agent(
    llm=llm,
    tools=[internet_search],
    system_prompt=research_instructions,
)

# Invoke agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "What is langgraph?"}]
})

# Access results
print(result["final_answer"])
print(result["confidence_score"])
```

## Key Features

### 1. LLM-Controlled Iteration

- **Intelligent Decision Making**: LLM decides when to stop iterating based on evidence quality
- **No Hard-Coded Rules**: Avoids arbitrary thresholds for confidence scores
- **Adaptive Behavior**: Adjusts iteration depth based on question complexity and evidence availability
- **Quality Over Quantity**: Focuses on meaningful improvements rather than fixed iteration counts

### 2. Problem Decomposition

### 2. Problem Decomposition

- Breaks complex questions into manageable sub-problems
- Each sub-problem is independently testable
- Hierarchical problem structure

### 3. Hypothesis Generation

### 3. Hypothesis Generation

- Creates multiple competing hypotheses
- Each hypothesis is falsifiable
- Based on gathered evidence

### 4. Iterative Verification

### 4. Iterative Verification

- Tests hypotheses using available tools
- Learns from both successes and failures
- Accumulates verified knowledge in experience pool

### 5. Experience Pool

### 5. Experience Pool

- Stores validated knowledge across iterations
- Informs future hypothesis generation
- Builds cumulative understanding

### 6. Confidence Scoring

### 6. Confidence Scoring

- LLM provides confidence assessment for each hypothesis
- Aggregate scoring based on accepted hypotheses
- Transparent uncertainty handling

## Tool Integration

The agent is tool-agnostic and can work with:

- Web search tools (DuckDuckGo, Google, etc.)
- Academic search tools
- Fact-checking tools
- Custom domain-specific tools

Tools are used in:

- Information gathering phase
- Hypothesis verification phase

## Configuration

### System Prompt

Customizable instructions for the agent's behavior and research approach. Should emphasize:

- Research thoroughness vs. efficiency balance
- When to continue iterating vs. when to finish
- Quality standards for evidence and hypotheses

### Max Iterations

Sets the hard upper limit for iterations (default: 3). The LLM will typically stop earlier based on evidence quality, but this prevents infinite loops.

### Tools

List of available tools for information gathering and verification.

## Output Structure

```python
{
    "original_question": str,
    "decomposed_problems": List[str],
    "gathered_information": List[Dict],
    "hypotheses": List[Hypothesis],
    "experience_pool": List[Dict],
    "final_answer": str,
    "current_iteration": int
}
```

## Implementation Status

**Status**: Fully implemented with LLM-controlled iteration

### Completed

- ✅ State definition with tool/LLM passing
- ✅ Node function implementations
- ✅ LLM-controlled iteration decision
- ✅ Agent creation interface
- ✅ Prompt templates (including should_continue prompt)
- ✅ Utility functions
- ✅ Example code with LLM iteration demo
- ✅ Documentation

### Pending

- ⏳ Comprehensive test implementations
- ⏳ Real-world validation with various LLMs
- ⏳ Performance optimization
- ⏳ Error handling improvements

## Future Enhancements

1. **Adaptive Iteration**: Dynamic iteration count based on hypothesis quality
2. **Parallel Hypothesis Testing**: Test multiple hypotheses concurrently
3. **Knowledge Graph**: Structure experience pool as a knowledge graph
4. **Meta-Learning**: Learn from past research sessions
5. **Collaborative Research**: Multi-agent research workflows

## References

- Popper, K. (1959). *The Logic of Scientific Discovery*
- Ries, E. (2011). *The Lean Startup*
- LangGraph Documentation: https://langchain-ai.github.io/langgraph/
