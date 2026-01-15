import json
import re
from typing import Optional
import argparse
from openai import OpenAI, AzureOpenAI
from litellm import batch_completion, _turn_on_debug, ModelResponse
from loguru import logger


VERIFICATION_PROMPT_TEMPLATE = """
Make a verification on whether the following response to the question is correct or not based on the ground truth answer.

Here is the question: 
<question>
{question}
</question>

Here is the response:
<response>
{response}
</response>

Here is the ground truth answer:
<ground_truth_answer>
{ground_truth_answer}
</ground_truth_answer>


Criteria for correctness:
- When the exact answer in the response is the same as the ground truth answer regardless of the format, the response is correct.
- For numerical answers, the response is correct if the exact answer in the response is within a small margin of the ground truth answer.

You should first think carefully for the correctness of the response.

You should output the verification result in the following format, and the result should be one of ["correct", "incorrect"]:
<verification_result>
( your verification result here, should be correct or incorrect )
</verification_result>
""".strip()


def get_content_from_tag(
    content: str,
    tag: str,
    default_value: Optional[str] = None
) -> Optional[str]:
    """
    extract the content from the first specified tag
    """
    if not content:
        return default_value

    pattern = rf"<{tag}>(.*?)(?=(</{tag}>|<\w+|$))"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return default_value


def main():
    dev_gt = json.load(open("/workspace/data/datasets/dev.json"))
    dev_pred = json.load(open("/workspace/data/outputs/dev.json"))

    batch_size = 10
    correct_count = 0

    for i in range(0, len(dev_gt), batch_size):
        batch_gt = dev_gt[i:i+batch_size]
        batch_pred = dev_pred[i:i+batch_size]

        batch_question = [item_gt["prompt"] for item_gt in batch_gt]

        batch_gt = [item_gt["answer"] for item_gt in batch_gt]
        batch_pred = [item_pred["answer"] for item_pred in batch_pred]

        batch_prompt = [VERIFICATION_PROMPT_TEMPLATE.format(
            question=question,
            response=response,
            ground_truth_answer=gt
        ) for question, response, gt in zip(batch_question, batch_pred, batch_gt)]

        batch_response = batch_completion(
            model="azure/o3", # o3 or azure/o3
            messages=[[{"role": "user", "content": prompt}] for prompt in batch_prompt],
            temperature=1, 
            max_tokens=1000,
            api_key="",
            api_version="",
            api_base="",
        )

        for res in batch_response:
            if res and isinstance(res, ModelResponse):
                verification_result = get_content_from_tag(res.choices[0].message.content, "verification_result")
                if verification_result:
                    if "correct" in verification_result.lower() and "incorrect" not in verification_result.lower():
                        correct_count += 1
            else:
                raise ValueError(f"Error in verification: {res}")
                

    print(f"Accuracy: {correct_count / len(dev_gt)}")



if __name__ == "__main__":
    main()