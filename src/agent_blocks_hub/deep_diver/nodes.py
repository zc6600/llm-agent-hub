"""
Node functions for the Deep Diver agent graph.

Each function represents a step in the scientific method.
"""

from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from .state import DeepDiverState
from .prompts import (
    FORMULATE_PROBLEM_PROMPT,
    GATHER_INFORMATION_PROMPT,
    GENERATE_HYPOTHESIS_PROMPT,
    VERIFY_HYPOTHESIS_PROMPT,
    FINAL_ANSWER_PROMPT,
    SHOULD_CONTINUE_PROMPT
)
from .utils import extract_user_question
from .task_classifier import TaskClassifier
import json


def classify_task(state: DeepDiverState) -> Dict[str, Any]:
    """
    Node 0: Classify the task to determine workflow complexity.
    
    Determines whether the task is "simple" (factual lookup) or "complex"
    (requires hypothesis generation and verification).
    
    Takes into account available tools when making the classification decision.
    
    Args:
        state: Current state of the agent
        
    Returns:
        Updated state with task classification and use_hypothesis_generation flag
    """
    llm = state["llm"]
    tools = state.get("tools", [])
    user_question = extract_user_question(state["messages"])
    
    print(f"\n[CLASSIFY] Analyzing task complexity")
    print(f"[CLASSIFY] Question: {user_question}")
    print(f"[CLASSIFY] Available tools: {[tool.name for tool in tools] if tools else ['None']}")
    
    # Get or create task classifier
    classifier = state.get("task_classifier")
    if classifier is None:
        classifier = TaskClassifier(llm=llm, tools=tools)
    
    # Classify the task (pass tools for consideration)
    classification = classifier.classify(user_question, task_type="auto", tools=tools)
    
    print(f"[CLASSIFY] Task Type: {classification['task_type']}")
    print(f"[CLASSIFY] Confidence: {classification['confidence']:.2f}")
    print(f"[CLASSIFY] Reasoning: {classification['reasoning']}")
    print(f"[CLASSIFY] Tool Consideration: {classification['tool_consideration']}")
    print(f"[CLASSIFY] Use Hypothesis Generation: {classification['use_hypothesis']}\n")
    
    return {
        "task_type": classification["task_type"],
        "task_classification_confidence": classification["confidence"],
        "task_reasoning": classification["reasoning"],
        "use_hypothesis_generation": classification["use_hypothesis"]
    }


def formulate_problem(state: DeepDiverState) -> Dict[str, Any]:
    """
    Node 1: Formulate and decompose the problem.
    
    Takes the user's question and breaks it down into manageable sub-problems.
    Inspired by Popper's problem-oriented approach.
    
    Args:
        state: Current state of the agent
        
    Returns:
        Updated state with decomposed problems
    """
    llm = state["llm"]
    system_prompt = state.get("system_prompt", "")
    
    # Extract user question
    user_question = extract_user_question(state["messages"])
    print(f"\n[FORMULATE] Extracted question: {user_question}")
    
    # Create prompt for problem decomposition
    prompt = FORMULATE_PROBLEM_PROMPT.format(question=user_question)
    
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))
    
    # Get LLM response
    response = llm.invoke(messages)
    print(f"[FORMULATE] LLM response (first 1000 chars):\n{response.content[:1000]}\n...")
    
    # Parse decomposed problems from response - expect JSON format
    decomposed = []
    try:
        import re
        content = response.content.strip()
        
        # Try to extract JSON object
        if "{" in content and "}" in content:
            # Find the JSON object
            start = content.index("{")
            end = content.rindex("}") + 1
            json_str = content[start:end]
            json_obj = json.loads(json_str)
            
            # Extract sub_problems array
            if "sub_problems" in json_obj:
                sub_problems = json_obj["sub_problems"]
                if isinstance(sub_problems, list):
                    # Extract problem text from list of objects
                    decomposed = [
                        item.get("problem", str(item)) if isinstance(item, dict) else str(item)
                        for item in sub_problems
                    ]
                    print(f"[FORMULATE] ✓ Parsed JSON structure: {len(decomposed)} problems")
            
        # Fallback: Parse as numbered list
        if not decomposed:
            pattern = r'^[0-9]+[.)]\s+(.+?)$'
            matches = re.finditer(pattern, content, re.MULTILINE)
            decomposed = [match.group(1).strip() for match in matches]
            if decomposed:
                print(f"[FORMULATE] ✓ Parsed as numbered list: {len(decomposed)} problems")
        
        # Final fallback: split by newlines
        if not decomposed:
            decomposed = [
                line.strip() 
                for line in content.split("\n")
                if line.strip() and len(line.strip()) > 10
            ][:5]
            if decomposed:
                print(f"[FORMULATE] ⚠ Fallback to newline parsing: {len(decomposed)} problems")
            
    except Exception as e:
        # Exception fallback
        print(f"[FORMULATE] ✗ Exception during JSON parsing: {e}")
        decomposed = []
    
    # Ensure we have at least the original question if parsing failed completely
    if not decomposed:
        print(f"[FORMULATE] ⚠ No problems extracted, using original question as single problem")
        decomposed = [user_question]
    
    print(f"[FORMULATE] Final decomposed problems ({len(decomposed)}):\n{json.dumps(decomposed, ensure_ascii=False, indent=2)}\n")
    
    return {
        "original_question": user_question,
        "decomposed_problems": decomposed,
        "current_iteration": 0,
        "gathered_information": [],
        "hypotheses": [],
        "experience_pool": []
    }


def gather_information(state: DeepDiverState) -> Dict[str, Any]:
    """
    Node 2: Gather information from available tools - PARALLEL VERSION.
    
    Uses provided tools to collect relevant information for each decomposed problem
    IN PARALLEL. Each problem gets its own research query through the LLM.
    
    Args:
        state: Current state of the agent
        
    Returns:
        Updated state with gathered information organized by problem
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    llm = state["llm"]
    tools = state.get("tools", [])
    system_prompt = state.get("system_prompt", "")
    decomposed_problems = state.get("decomposed_problems", [])
    original_question = state.get("original_question", "")
    
    print(f"\n[GATHER] Starting PARALLEL information gathering")
    print(f"[GATHER] Original question: {original_question}")
    print(f"[GATHER] Decomposed problems to research: {len(decomposed_problems)}")
    print(f"[GATHER] Available tools: {[tool.name for tool in tools] if tools else ['None']}")
    
    # Create LLM with tools
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    tool_names = [tool.name for tool in tools] if tools else ["No tools available"]
    
    def research_single_problem(problem_idx: int, problem: str) -> Dict[str, Any]:
        """Research a single decomposed problem - executed in parallel."""
        print(f"[GATHER] [{problem_idx+1}/{len(decomposed_problems)}] Researching: {problem[:60]}...")
        
        # Simple, direct research prompt
        research_prompt = f"""Research the following sub-problem related to: {original_question}

SUB-PROBLEM: {problem}

Use available tools to find relevant information. Be concise and direct."""
        
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=research_prompt))
        
        # Invoke LLM with tools
        response = llm_with_tools.invoke(messages)
        
        # Collect tool calls and results
        problem_info = []
        llm_response = response.content
        
        # Check if there are tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            # We'll collect the tool results but NOT invoke synthesis
            # The synthesis happens in the synthesize_results node
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                
                # Find and execute the tool
                for tool in tools:
                    if tool.name == tool_name:
                        try:
                            result = tool.invoke(tool_args)
                            problem_info.append({
                                "tool": tool_name,
                                "query": tool_args,
                                "result": result,
                                "problem_idx": problem_idx,
                                "problem": problem
                            })
                            print(f"[GATHER] [{problem_idx+1}] ✓ {tool_name}: {str(result)[:80]}...")
                        except Exception as e:
                            problem_info.append({
                                "tool": tool_name,
                                "query": tool_args,
                                "error": str(e),
                                "problem_idx": problem_idx,
                                "problem": problem
                            })
                            print(f"[GATHER] [{problem_idx+1}] ✗ Tool error: {str(e)[:50]}")
                        break
        
        return {
            "problem_idx": problem_idx,
            "problem": problem,
            "llm_response": llm_response,  # LLM's initial analysis
            "gathered_info": problem_info   # Tool results
        }
    
    # Execute all problem researches in parallel
    gathered_info = []
    problem_results = []
    
    with ThreadPoolExecutor(max_workers=min(3, len(decomposed_problems))) as executor:
        # Submit all tasks
        futures = {
            executor.submit(research_single_problem, idx, problem): idx
            for idx, problem in enumerate(decomposed_problems)
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                result = future.result()
                problem_results.append(result)
                gathered_info.extend(result["gathered_info"])
            except Exception as e:
                idx = futures[future]
                print(f"[GATHER] [{idx+1}] ✗ Exception: {str(e)}")
    
    # Sort results by problem index to maintain order
    problem_results.sort(key=lambda x: x["problem_idx"])
    
    print(f"\n[GATHER] ✓ Completed parallel research of {len(problem_results)} problems")
    print(f"[GATHER] Total tool results gathered: {len(gathered_info)}\n")
    
    # Merge with existing gathered information
    existing_info = state.get("gathered_information", [])
    all_info = existing_info + gathered_info
    
    return {
        "gathered_information": all_info,
        "problem_research_results": problem_results  # Store structured results for synthesis
    }


def generate_hypothesis(state: DeepDiverState) -> Dict[str, Any]:
    """
    Node 3: Generate hypotheses based on gathered information.
    
    Creates testable hypotheses that could answer the problem.
    Follows Popper's falsificationism principle.
    
    Args:
        state: Current state of the agent
        
    Returns:
        Updated state with generated hypotheses
    """
    llm = state["llm"]
    system_prompt = state.get("system_prompt", "")
    original_question = state.get("original_question", "")
    gathered_info = state.get("gathered_information", [])
    experience_pool = state.get("experience_pool", [])
    
    print(f"\n[HYPOTHESIS_GEN] Generating hypotheses")
    print(f"[HYPOTHESIS_GEN] Original question: {original_question}")
    print(f"[HYPOTHESIS_GEN] Gathered information sources: {len(gathered_info)}")
    print(f"[HYPOTHESIS_GEN] Experience pool entries: {len(experience_pool)}")
    
    # Format gathered information
    info_text = "\n\n".join([
        f"Source: {info.get('tool', 'unknown')}\n"
        f"Query: {info.get('query', 'N/A')}\n"
        f"Result: {info.get('result', info.get('error', 'No result'))}"
        for info in gathered_info
    ])
    
    # Include experience pool context
    if experience_pool:
        exp_text = "\n".join([
            f"- {exp.get('hypothesis', '')}: {exp.get('result', '')}"
            for exp in experience_pool
        ])
        info_text += f"\n\nPrevious findings:\n{exp_text}"
    
    # Create prompt for hypothesis generation
    prompt = GENERATE_HYPOTHESIS_PROMPT.format(
        question=original_question,
        information=info_text or "No information gathered yet"
    )
    
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))
    
    # Get LLM response
    response = llm.invoke(messages)
    print(f"[HYPOTHESIS_GEN] LLM response (first 300 chars):\n{response.content[:300]}\n...")
    
    # Parse hypotheses from response
    try:
        import re
        content = response.content.strip()
        
        # Try to extract JSON object
        if "{" in content and "}" in content:
            # Find the JSON object
            start = content.index("{")
            end = content.rindex("}") + 1
            json_str = content[start:end]
            json_obj = json.loads(json_str)
            
            # Extract hypotheses array
            if "hypotheses" in json_obj:
                hypotheses_list = json_obj["hypotheses"]
                if isinstance(hypotheses_list, list):
                    new_hypotheses = [
                        {
                            "content": item.get("hypothesis", str(item)),
                            "verification_result": None,
                            "confidence": None,
                            "evidence": []
                        }
                        for item in hypotheses_list
                    ]
                    print(f"[HYPOTHESIS_GEN] ✓ Parsed JSON structure: {len(new_hypotheses)} hypotheses")
        
        # Fallback: Parse as numbered list
        if not new_hypotheses:
            pattern = r'^[0-9]+[.)]\s+(.+?)$'
            matches = re.finditer(pattern, content, re.MULTILINE)
            new_hypotheses = [
                {
                    "content": match.group(1).strip(),
                    "verification_result": None,
                    "confidence": None,
                    "evidence": []
                }
                for match in matches
            ][:4]  # Limit to 4 hypotheses
            
            if new_hypotheses:
                print(f"[HYPOTHESIS_GEN] ✓ Parsed numbered list: {len(new_hypotheses)} hypotheses")
        
        # Final fallback: use entire response as one hypothesis
        if not new_hypotheses:
            print(f"[HYPOTHESIS_GEN] ⚠ No structured hypotheses found, using entire response")
            new_hypotheses = [{
                "content": content.strip(),
                "verification_result": None,
                "confidence": None,
                "evidence": []
            }]
            
    except Exception as e:
        # Exception fallback: use entire response as one hypothesis
        print(f"[HYPOTHESIS_GEN] ✗ Exception during parsing: {e}")
        new_hypotheses = [{
            "content": response.content.strip(),
            "verification_result": None,
            "confidence": None,
            "evidence": []
        }]
    
    print(f"[HYPOTHESIS_GEN] Final hypotheses ({len(new_hypotheses)}):\n{json.dumps([h['content'] for h in new_hypotheses], ensure_ascii=False, indent=2)}\n")
    
    return {
        "hypotheses": new_hypotheses
    }


def verify_hypothesis(state: DeepDiverState) -> Dict[str, Any]:
    """
    Node 4: Verify hypotheses and add to experience pool.
    
    Tests each hypothesis using available tools and evidence.
    Successful verifications are added to the experience pool.
    Inspired by Lean Startup's Build-Measure-Learn cycle.
    
    Args:
        state: Current state of the agent
        
    Returns:
        Updated state with verification results and updated experience pool
    """
    llm = state["llm"]
    tools = state.get("tools", [])
    system_prompt = state.get("system_prompt", "")
    hypotheses = state.get("hypotheses", [])
    gathered_info = state.get("gathered_information", [])
    experience_pool = state.get("experience_pool", [])
    current_iteration = state.get("current_iteration", 0)
    
    print(f"\n[VERIFY] Verifying hypotheses")
    print(f"[VERIFY] Hypotheses to verify: {len(hypotheses)}")
    print(f"[VERIFY] Current iteration: {current_iteration}")
    
    # Create LLM with tools
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    
    verified_hypotheses = []
    
    # Verify each hypothesis
    for hypothesis in hypotheses:
        hyp_content = hypothesis["content"]
        
        # Format evidence
        evidence_text = "\n\n".join([
            f"Source: {info.get('tool', 'unknown')}\n"
            f"Result: {info.get('result', info.get('error', 'No result'))}"
            for info in gathered_info
        ])
        
        tool_names = [tool.name for tool in tools] if tools else ["No tools available"]
        
        # Create verification prompt
        prompt = VERIFY_HYPOTHESIS_PROMPT.format(
            hypothesis=hyp_content,
            evidence=evidence_text or "No evidence available yet",
            tool_names=", ".join(tool_names)
        )
        
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        # Get verification
        response = llm_with_tools.invoke(messages)
        
        # Execute any tool calls for additional verification
        additional_evidence = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                
                for tool in tools:
                    if tool.name == tool_name:
                        try:
                            result = tool.invoke(tool_args)
                            additional_evidence.append(result)
                        except Exception as e:
                            additional_evidence.append(f"Error: {str(e)}")
                        break
        
        # Parse verification result and confidence
        verification_content = response.content
        
        # Extract confidence score (look for patterns like "confidence: 0.8" or "80%")
        confidence = 0.5  # Default
        try:
            import re
            conf_match = re.search(r'confidence[:\s]+([0-9.]+)', verification_content.lower())
            if conf_match:
                conf_val = float(conf_match.group(1))
                confidence = conf_val if conf_val <= 1.0 else conf_val / 100.0
            else:
                percent_match = re.search(r'([0-9]+)%', verification_content)
                if percent_match:
                    confidence = float(percent_match.group(1)) / 100.0
        except Exception:
            pass
        
        # Determine verification result
        verification_result = "accepted" if confidence >= 0.6 else "rejected" if confidence < 0.4 else "uncertain"
        
        verified_hyp = {
            "content": hyp_content,
            "verification_result": verification_result,
            "confidence": confidence,
            "evidence": additional_evidence
        }
        
        verified_hypotheses.append(verified_hyp)
        
        # Add to experience pool if accepted or interesting
        if verification_result in ["accepted", "uncertain"]:
            experience_pool.append({
                "hypothesis": hyp_content,
                "result": verification_result,
                "confidence": confidence,
                "iteration": current_iteration,
                "evidence": additional_evidence
            })
    
    return {
        "hypotheses": verified_hypotheses,
        "experience_pool": experience_pool,
        "current_iteration": current_iteration + 1
    }


def final_answer(state: DeepDiverState) -> Dict[str, Any]:
    """
    Node 5: Generate final answer based on all research and analysis.
    
    Synthesizes all verified hypotheses, experience pool, and synthesized research results
    into a comprehensive answer to the original question.
    
    Args:
        state: Current state of the agent
        
    Returns:
        Updated state with final answer
    """
    llm = state["llm"]
    system_prompt = state.get("system_prompt", "")
    original_question = state.get("original_question", "")
    decomposed_problems = state.get("decomposed_problems", [])
    hypotheses = state.get("hypotheses", [])
    experience_pool = state.get("experience_pool", [])
    synthesized_research = state.get("synthesized_research", [])
    synthesis_text = state.get("synthesis_text", "")
    
    print(f"\n[FINAL_ANSWER] Building comprehensive answer")
    print(f"[FINAL_ANSWER] Original question: {original_question}")
    print(f"[FINAL_ANSWER] Decomposed problems: {len(decomposed_problems)}")
    print(f"[FINAL_ANSWER] Synthesized research available: {bool(synthesis_text)}")
    
    # Build answer context from available sources
    answer_context_parts = []
    
    # Always show the decomposed problems
    if decomposed_problems:
        problems_text = "\n".join([
            f"{i+1}. {p}"
            for i, p in enumerate(decomposed_problems)
        ])
        answer_context_parts.append(f"DECOMPOSED RESEARCH QUESTIONS:\n{problems_text}")
    
    # Include synthesized research if available (from parallel gathering)
    if synthesis_text:
        answer_context_parts.append(f"SYNTHESIZED RESEARCH FINDINGS:\n{synthesis_text}")
    
    # Include verified hypotheses if available (from hypothesis verification path)
    if hypotheses:
        verified_text = "\n\n".join([
            f"Hypothesis {i+1}: {h.get('content', '')}\n"
            f"Status: {h.get('verification_result', 'unknown')}\n"
            f"Confidence: {h.get('confidence', 0.0):.2f}\n"
            f"Evidence: {', '.join(str(e) for e in h.get('evidence', [])) or 'None'}"
            for i, h in enumerate(hypotheses)
        ])
        answer_context_parts.append(f"VERIFIED HYPOTHESES:\n{verified_text}")
    
    # Include experience pool if available
    if experience_pool:
        exp_text = "\n".join([
            f"- [{exp.get('result', 'unknown')}] {exp.get('hypothesis', '')} "
            f"(confidence: {exp.get('confidence', 0.0):.2f}, iteration: {exp.get('iteration', 0)})"
            for exp in experience_pool
        ])
        answer_context_parts.append(f"EXPERIENCE POOL (Previous Findings):\n{exp_text}")
    
    # Combine all context
    combined_context = "\n\n".join(answer_context_parts) if answer_context_parts else "No detailed research available"
    
    # Create final answer prompt
    final_prompt = f"""
Based on comprehensive research and analysis, provide a final, well-integrated answer to the original question.

ORIGINAL QUESTION: {original_question}

RESEARCH AND ANALYSIS CONTEXT:
{combined_context}

INSTRUCTIONS:
1. Review all the research findings and verified hypotheses above
2. Synthesize them into a cohesive, comprehensive final answer
3. Integrate findings from all researched perspectives
4. Provide clear reasoning for your conclusions
5. Note any limitations, uncertainties, or gaps in the research
6. Give a clear, actionable final answer

Generate a final answer that:
- Directly addresses the original question
- Integrates all findings logically
- Clearly states the conclusion
- Explains the reasoning and evidence
- Acknowledges confidence levels and uncertainties
"""
    
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=final_prompt))
    
    # Get final answer (THIS WILL BE MONITORED BY LANGSMITH)
    response = llm.invoke(messages)
    
    print(f"\n[FINAL_ANSWER] ✓ Generated comprehensive final answer")
    print(f"[FINAL_ANSWER] Answer length: {len(response.content)} chars\n")

    return {
        "final_answer": response.content,
        "messages": state["messages"] + [AIMessage(content=response.content)]
    }


def should_continue_iteration(state: DeepDiverState) -> str:
    """
    Conditional edge: Determine if iteration should continue.
    
    Uses LLM to intelligently decide whether to continue or finish based on
    the quality and completeness of verified hypotheses.
    
    Args:
        state: Current state of the agent
        
    Returns:
        "continue" to iterate again, "finish" to proceed to final answer
    """
    llm = state["llm"]
    system_prompt = state.get("system_prompt", "")
    current_iteration = state.get("current_iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    original_question = state.get("original_question", "")
    hypotheses = state.get("hypotheses", [])
    experience_pool = state.get("experience_pool", [])
    
    # Hard limit: if max iterations reached, must finish
    if current_iteration >= max_iterations:
        return "finish"
    
    # If no hypotheses yet (first iteration), continue
    if not hypotheses:
        return "continue"
    
    # Format hypotheses summary
    hypotheses_summary = "\n".join([
        f"  {i+1}. [{h.get('verification_result', 'unknown')}] "
        f"{h.get('content', '')[:100]}... "
        f"(confidence: {h.get('confidence', 0.0):.2f})"
        for i, h in enumerate(hypotheses)
    ])
    
    # Format experience pool
    exp_text = "\n".join([
        f"  - [{exp.get('result', 'unknown')}] {exp.get('hypothesis', '')[:80]}... "
        f"(conf: {exp.get('confidence', 0.0):.2f})"
        for exp in experience_pool
    ]) if experience_pool else "  (empty)"
    
    # Ask LLM to decide
    prompt = SHOULD_CONTINUE_PROMPT.format(
        current_iteration=current_iteration,
        max_iterations=max_iterations,
        question=original_question,
        num_hypotheses=len(hypotheses),
        hypotheses_summary=hypotheses_summary or "  (none)",
        experience_pool=exp_text
    )
    
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))
    
    # Get LLM decision
    response = llm.invoke(messages)
    decision = response.content.strip().upper()
    
    # Parse decision
    if "CONTINUE" in decision:
        return "continue"
    elif "FINISH" in decision:
        return "finish"
    else:
        # Default: if can't parse, continue if below max, else finish
        return "continue" if current_iteration < max_iterations - 1 else "finish"


def decide_hypothesis_needed(state: DeepDiverState) -> str:
    """
    Conditional edge: Determine if hypothesis generation is needed.
    
    Routes to either hypothesis generation (complex path) or directly to
    final answer (simple path) based on task classification.
    
    Args:
        state: Current state of the agent
        
    Returns:
        "hypothesis" to proceed with hypothesis generation, "answer" to skip to final answer
    """
    use_hypothesis = state.get("use_hypothesis_generation", False)
    task_type = state.get("task_type", "complex")
    
    print(f"\n[ROUTE] Deciding workflow path")
    print(f"[ROUTE] Task Type: {task_type}")
    print(f"[ROUTE] Use Hypothesis Generation: {use_hypothesis}")
    
    if use_hypothesis:
        print(f"[ROUTE] → Taking COMPLEX path (with hypothesis generation)\n")
        return "hypothesis"
    else:
        print(f"[ROUTE] → Taking SIMPLE path (direct to final answer)\n")
        return "answer"


def synthesize_results(state: DeepDiverState) -> Dict[str, Any]:
    """
    Node: Synthesize results from parallel problem research using LLM.
    
    Takes the individual research results from each decomposed problem,
    asks the LLM to synthesize them into a cohesive overview,
    and prepares the data for final answer generation.
    
    This node can be monitored by LangSmith properly.
    
    Args:
        state: Current state of the agent
        
    Returns:
        Updated state with synthesized research overview
    """
    llm = state["llm"]
    system_prompt = state.get("system_prompt", "")
    problem_results = state.get("problem_research_results", [])
    original_question = state.get("original_question", "")
    
    print(f"\n[SYNTHESIZE] Synthesizing results from {len(problem_results)} problem researches")
    print(f"[SYNTHESIZE] Original question: {original_question}")
    
    # Build context from all problem researches
    problem_summaries = []
    for result in problem_results:
        problem = result.get("problem", "")
        llm_response = result.get("llm_response", "")
        gathered_info = result.get("gathered_info", [])
        
        # Collect tool results
        tool_results = []
        for info in gathered_info:
            tool_name = info.get("tool", "unknown")
            tool_result = info.get("result", info.get("error", ""))
            tool_results.append(f"[{tool_name}] {tool_result}")
        
        problem_summaries.append({
            "problem": problem,
            "llm_initial_response": llm_response,
            "tool_results": tool_results,
            "num_sources": len(gathered_info)
        })
    
    # Create synthesis prompt for LLM
    synthesis_context = "\n\n".join([
        f"PROBLEM {i+1}: {s['problem']}\n"
        f"LLM Analysis: {s['llm_initial_response'][:200]}...\n"
        f"Tool Results ({s['num_sources']} sources):\n" +
        "\n".join(f"  - {r}" for r in s['tool_results'][:3])
        for i, s in enumerate(problem_summaries)
    ])
    
    synthesis_prompt = f"""
You are tasked with synthesizing research findings into a comprehensive overview.

ORIGINAL QUESTION: {original_question}

RESEARCH FINDINGS FROM MULTIPLE SUB-PROBLEMS:
{synthesis_context}

Please provide a coherent synthesis that:
1. Integrates findings from all sub-problems
2. Identifies key themes and connections
3. Highlights the most important discoveries
4. Notes any gaps or uncertainties

SYNTHESIS:"""
    
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=synthesis_prompt))
    
    # Get LLM synthesis (THIS WILL BE MONITORED BY LANGSMITH)
    response = llm.invoke(messages)
    synthesis_text = response.content
    
    # Create structured synthesis overview
    synthesis_overview = []
    for idx, result in enumerate(problem_results):
        problem = result.get("problem", "")
        gathered_info = result.get("gathered_info", [])
        
        synthesis_overview.append({
            "problem_idx": idx + 1,
            "problem": problem,
            "llm_response": result.get("llm_response", ""),
            "num_sources": len(gathered_info),
            "sources": [
                {
                    "tool": info.get("tool"),
                    "query": info.get("query"),
                    "snippet": str(info.get("result", info.get("error", "")))[:200]
                }
                for info in gathered_info[:3]
            ]
        })
    
    print(f"\n[SYNTHESIZE] ✓ Synthesized overview of all {len(synthesis_overview)} problem researches")
    print(f"[SYNTHESIZE] Synthesis preview:\n{synthesis_text[:300]}...\n")
    
    return {
        "synthesized_research": synthesis_overview,
        "synthesis_text": synthesis_text
    }
