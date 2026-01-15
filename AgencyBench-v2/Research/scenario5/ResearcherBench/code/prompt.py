#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prompt templates for AI model response evaluation
"""

def get_single_rubric_evaluation_prompt(question: str, rubric: str, weight: int, 
                                          ai_response: str) -> str:
    """
    Returns a prompt for evaluating if a single criterion is covered in the AI response
    
    Parameters:
    question (str): The research question posed by the user
    rubric (str): The criterion content to be evaluated
    rubric_weight (int): The weight of the criterion, indicating its importance
    ai_response (str): The complete AI response text
    
    Returns:
    str: Prompt for evaluating a single criterion
    """
    prompt = f"""# Rubric Coverage Evaluation

## Task
Determine whether the AI response adequately covers the specific criterion provided. Answer with "yes" or "no" followed by a brief justification.

## Input Materials
<Question>: {question}
<criterion>: {rubric}
<Weight>: {weight} (indicates the importance of this criterion)
<AI Response>: {ai_response}

## Evaluation Criteria
- Answer "yes" if the AI response clearly includes or adequately expresses the main content of the criterion
- Answer "yes" if the response conveys the same meaning as the criterion, even if using different terminology or phrasing
- Answer "no" if the AI response only partially addresses or completely fails to mention the content of the criterion
- Consider semantic equivalence, not just keyword matching
- Pay special attention to technical details, numerical values, and specific claims in the criterion

## Output Format
Your answer must begin with either "yes" or "no" followed by a brief justification.

Example format:
"yes: The response clearly addresses this criterion by explaining [specific detail]..."
"no: While the response mentions [related concept], it fails to address [specific aspect] of the criterion..."

Note: Your assessment will be used to calculate recall metrics, so accuracy is critical."""

    return prompt


EXTRACT_CLAIMS_PROMPT = """
## Task Description
Extract all factual claims from the provided academic paper. Each claim should be a factual statement that can be verified. Claims may or may not have supporting citations.

## Input
A Research Question and a complete academic paper containing factual claims, some of which may have citation markers and corresponding URLs (either inline or in a reference section).

## Output Requirements
- Extract each distinct factual claim throughout the entire paper
- For each claim, output a JSON object with:
  - The exact claim text as a string
  - The original text from the paper containing this claim (context)
  - The corresponding citation URL as source (if a citation marker directly follows the claim)
- If a claim has a citation marker directly following it, return the supporting URL as source
- If a claim does not have a citation marker directly following it, return an empty string for source
- Ensure all string values are properly escaped for valid JSON format (e.g. Replace internal quotation marks (") with escaped quotation marks (\")) in the claim and context
- Return a JSON array containing all claim objects

## Format Specification
```json
[
  {
    "claim": "The exact statement representing a factual claim",
    "context": "The original sentence or passage from the paper containing this claim",
    "source": "https://example.com/source1"
  },
  {
    "claim": "Another factual statement without direct citation",
    "context": "The original sentence or passage from the paper containing this claim",
    "source": ""
  }
]
```

## Guidelines for Claim Identification
1. A claim should be a complete, standalone factual statement
2. Maintain the original wording where possible, but remove unnecessary context
3. Extract all factual claims regardless of whether they have citation support
4. Only consider to map citation markers (numbers, author names, etc.) to their corresponding URLs in the references section when it directly follow the claim statement.
5. Exclude opinions, speculations, or methodological descriptions
6. Extract the context passage containing each claim for verification purposes
7. If multiple claims are associated with the same citation, extract them as separate entries


## Citation URL Mapping
- If URLs appear directly after claims, use those URLs directly
- Citation markers (e.g. follows a number or [number]) must directly follow the claim to be considered as supporting that claim
- If claims use citation markers that reference a bibliography or reference section, locate the corresponding URLs in that section
- If a claim has no directly following citation marker, use an empty string for source

---
Please extract all claims from the following paper and provide them in the specified JSON format:

Research Question: 
[QUESTION]

Response Content:
[CONTENT]

References:
[REFERENCES]

"""


# Prompt to verify a claim against source content
VERIFY_CLAIM_PROMPT = """
## Task Description
Your task is to verify whether multiple claims are supported by the provided reference content.

## Input
- A reference content that contains supporting information
- A list of claim-context pairs that need to be verified against the reference

## Output
For each claim, respond with 'yes', 'no', or 'unknown' to indicate whether the claim is supported by the reference content. Output in the specified JSON format.

## Output Format Specification
```json
[
  {
    "id": 1,
    "result": "yes"
  },
  {
    "id": 2,
    "result": "no"
  },
  {
    "id": 3,
    "result": "unknown"
  }
]
```

## Verification Guidelines

### Claim Support Determination
If the reference is valid, for each given claim:
- **'yes'**: If the facts or data in the claim can be found entirely or partially within the reference content
- **'no'**: If all facts and data in the statement cannot be found in the reference content
- **'unknown'**: If verification encounters difficulties (such as semantic incompleteness, ambiguity, or other issues that make verification impossible), or reference contains are not available ('page not found' message, connection errors, or other non-content responses).

Notice that claims must be verifiable from the content provided, not based on general knowledge.

### Using Context Information
If you encounter difficulties when verifying claims (e.g., semantic incompleteness/ambiguity issues), refer to the corresponding additional context. If problems still exist after considering the paragraph context, output 'unknown'.

---

Please provide your verification results in the specified JSON format.

Source:
[SOURCE]

Claim-Paragraph Pair List: 
[CLAIM_LIST]

"""

# System messages for different roles
CLAIM_EXTRACTOR_SYSTEM = "You are an expert research analyzer focused on identifying factual claims and their citations."
CLAIM_VERIFIER_SYSTEM = "You are an expert fact-checker focused on verifying claims against source material."

def generate_prompt(template, paras):
    """
    Generate a prompt by replacing placeholders with parameter values.
    
    Args:
        template (str): The prompt template with placeholders in [PARAM_NAME] format
        paras (dict): Dictionary of parameter names and values
        
    Returns:
        str: The prompt with placeholders replaced with parameter values
    """
    prompt = template
    for k in paras.keys():
        prompt = prompt.replace(f"[{k}]", str(paras[k]))
    return prompt 