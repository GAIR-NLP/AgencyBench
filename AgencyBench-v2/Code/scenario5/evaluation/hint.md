# A Basic Framework for Deep Research Agent

This document provides detailed guidance for implementing a basic deep research agent that can systematically research complex questions using web search and browsing tools.

## Core Architecture Overview

The research agent follows an **iterative decision-making approach** with the following key components:

- **Main Agent Loop**: Iterates up to 5 steps, making decisions at each step
- **Decision Making**: At each step, decides between "search_web" or "return_answer"
- **Web Search & Browsing**: Uses search engine tool and web browsing agent
- **Research Trajectory**: Maintains a complete history of all actions and results
- **Fallback Mechanism**: Forces a final answer if max steps reached

## Core Workflow Structure

### 1. Iterative Decision-Making Loop

**Implementation**: The agent runs a loop for up to `max_step = 5` iterations

**At Each Iteration**:
- Uses the `DECISION_MAKING_PROMPT_TEMPLATE` to analyze current state
- Reviews the research question and accumulated research trajectory
- Decides between two actions: `search_web` or `return_answer`
- Tracks all actions and results in the research trajectory

**Key Components**:
- **System Prompt**: Guides the agent to make research plans and solve sub-questions one at a time
- **Decision Prompt**: Contains the research question and full trajectory, asks for next action
- **Action Extraction**: Parses the LLM response to extract the chosen action

### 2. Web Search & Information Gathering

**When Action = "search_web"**:

**Search Query Generation**:
- Agent generates 1-3 search queries simultaneously
- Each query focuses on different aspects or sub-questions
- Queries are extracted from `<query>` tags in the LLM response
- Emphasizes concise, effective search terms

**Search Execution**:
- Uses `SearchEngineTool` to perform web searches
- Returns results with URL, title, and snippet for each query
- All search results are appended to the research trajectory

**Web Browsing**:
- Filters out already-browsed URLs using `self.browsed_urls` set
- Uses `WebBrowsingAgent` to extract useful content from web pages
- Passes research question context to focus content extraction
- Results are appended to research trajectory with URL and useful contents

**Trajectory Updates**:
```
[Agent Performs Web Searching]
Search Queries: [list of queries]
Search Result: 
<URL>...</URL>
<Title>...</Title>
<Snippet>...</Snippet>

[Agent Performs Web Browsing]
<URL>...</URL>
<Useful Contents>...</Useful Contents>
```

### 3. Answer Decision & Return

**When Action = "return_answer"**:
- Agent extracts answer from `<answer>` tags in LLM response
- Immediately yields the answer and breaks the loop
- No further processing or validation steps

**Fallback Answer Generation**:
- If max steps (5) reached without returning answer, triggers `FINAL_ANSWER_PROMPT_TEMPLATE`
- Forces the agent to provide best possible answer based on accumulated research
- Uses the complete research trajectory as context
- Returns answer even if incomplete: "Unable to find the answer" if no answer extracted

## Key Implementation Details

### Research Trajectory Management
- **Purpose**: Maintains complete history of all search queries, results, and browsing outcomes
- **Format**: Structured text with clear section markers and XML-like tags
- **Usage**: Passed to decision-making prompts to inform subsequent actions
- **Deduplication**: Tracks `browsed_urls` to avoid re-browsing same pages

### Prompt Structure
1. **System Prompt**: Brief guidance on research planning and sub-question approach
2. **Decision Prompt**: Contains research question + trajectory, asks for action choice
3. **Final Answer Prompt**: Backup prompt to force answer when max steps reached

### Action Parsing
- Uses utility functions `get_content_from_tag()` and `get_all_content_from_tag()`
- Extracts actions from `<action>` tags
- Extracts search queries from multiple `<query>` tags
- Extracts answers from `<answer>` tags

### Error Handling
- Invalid actions are logged to trajectory and loop continues
- Missing query content continues to next iteration
- No search results logged as "No search results found"
- No useful browsing content logged as "No useful contents found"

## Performance Requirements

In our implementation, this framework runs 1 hour for predictions in the dev set and got 0.56 accuracy. It runs 4.5 hours in the test set for predictions and got 0.55 accuracy. So you should at least match these results or do better.