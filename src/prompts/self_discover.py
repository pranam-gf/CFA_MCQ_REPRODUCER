"""
Prompt template for the Self-Discover reasoning strategy.

Guides the LLM to first outline a reasoning structure before solving the problem.
Inspired by: https://arxiv.org/abs/2402.14310v1
"""
from ..utils.prompt_utils import parse_question_data

SELF_DISCOVER_PROMPT_TEMPLATE = """\
**Task:** Solve the following multiple-choice question by first devising a reasoning structure using the Self-Discover method.

**Context/Vignette:**
{vignette}

**Question Stem:**
{question_stem}

**Options:**
A: {option_a}
B: {option_b}
C: {option_c}

**Self-Discover Reasoning Process:**

**1. Select Reasoning Modules:**
   Identify and list the core reasoning modules or types of thinking needed to solve this specific question. Examples: Causal Reasoning, Definition Understanding, Calculation, Comparison, Rule Application, Concept Identification, etc.

**2. Adapt Modules to the Problem:**
   For each selected module, briefly explain how it specifically applies to this question and the given options. Outline the steps you will take within each module.

**3. Implement Reasoning Structure:**
   Execute the plan outlined above step-by-step.
   [Model generates reasoning steps here]

**4. Final Answer:**
   Based on the reasoning, critically evaluate the options and provide the final answer.
   **IMPORTANT**: Conclude your response with the final answer choice letter (A, B, or C) on a new line, formatted exactly as: `The final answer is: **[Option Letter]**` (e.g., `The final answer is: **B**`). Do not include any other text after this final line.
   [Model provides final answer letter here in the specified format]

**Begin Reasoning:**

[Your reasoning structure and step-by-step solution here]

**Final Answer:** [Correct Option Letter]
"""

def format_self_discover_prompt(question_data: dict) -> str:
    """
    Formats the Self-Discover prompt with the specific question and options,
    using the standardized parsing utility.

    Args:
        question_data: Dictionary for a single MCQ item.

    Returns:
        Formatted prompt string.
    """
    parsed_data = parse_question_data(question_data)

    return SELF_DISCOVER_PROMPT_TEMPLATE.format(
        vignette=parsed_data['vignette'],
        question_stem=parsed_data['question_stem'],
        option_a=parsed_data['options_dict'].get('A', 'Option A not provided'),
        option_b=parsed_data['options_dict'].get('B', 'Option B not provided'),
        option_c=parsed_data['options_dict'].get('C', 'Option C not provided')
    )

def generate_prompt_for_self_discover_strategy(entry: dict) -> str:
    """Generates the Self-Discover prompt for the LLM using the standardized parser."""
    parsed_data = parse_question_data(entry) 
    return SELF_DISCOVER_PROMPT_TEMPLATE.format(
        vignette=parsed_data['vignette'],
        question_stem=parsed_data['question_stem'],
        option_a=parsed_data['options_dict'].get('A', 'Option A not provided'),
        option_b=parsed_data['options_dict'].get('B', 'Option B not provided'),
        option_c=parsed_data['options_dict'].get('C', 'Option C not provided')
    ) 