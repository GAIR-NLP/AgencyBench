import os
from typing import Dict, Any, Optional
import json

from evaluations.base.data_classes import Config
from evaluations.base.base_eval import BaseBenchmark
import re
from typing import Optional
from tqdm import tqdm

from litellm import batch_completion, ModelResponse


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



class TaskBenchmark(BaseBenchmark):
    """一个简单的基准测试示例"""

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self.output_dir_path = os.path.join(self.config.workspace, "data/outputs")
        self.reference_dir_path = os.path.join(self.config.eval_workspace, self.config.task_name, "data/references")
    
    def run_1(self):
        pred_data = json.load(open(f"{self.output_dir_path}/test.json"))
        gt_data = json.load(open(f"{self.reference_dir_path}/test.json"))

        if len(gt_data) != len(pred_data):
            return {
                "error": "Error: the length of gt_data and pred_data is not the same",
                "score": 0.0,
                "test_set_result": {
                    "error": "Error: the length of gt_data and pred_data is not the same",
                    "message": "Please check the pred_data",
                    "score": 0.0,
                }
            }
            
        batch_size = self.config.batch_size
        correct_count = 0

        for i in tqdm(range(0, len(gt_data), batch_size)):
            batch_gt = gt_data[i:i+batch_size]
            batch_pred = pred_data[i:i+batch_size]

            batch_question = [item_gt["prompt"] for item_gt in batch_gt]
            batch_gt = [item_gt["answer"] for item_gt in batch_gt]
            try:
                batch_pred = [item_pred["answer"] for item_pred in batch_pred]
            except:
                return {
                    "error": "Error: no `answer` key in pred_data",
                    "score": 0.0,
                    "test_set_result": {
                        "error":  "Error: no `answer` key in pred_data",
                        "message": "Please check the pred_data",
                        "score": 0.0,
                }
            }
            batch_prompt = [VERIFICATION_PROMPT_TEMPLATE.format(
                question=question,
                response=response,
                ground_truth_answer=gt
            ) for question, response, gt in zip(batch_question, batch_pred, batch_gt)]
            
            
            batch_response = batch_completion(
                model=self.config.model_name,
                messages=[[{"role": "user", "content": prompt}] for prompt in batch_prompt],
                temperature=1, 
                max_tokens=1000,
                api_key=self.config.api_key,
                api_version=self.config.api_version,
                api_base=self.config.api_endpoint,
            )

            for res in batch_response:
                if res and isinstance(res, ModelResponse):
                    verification_result = get_content_from_tag(res.choices[0].message.content, "verification_result")
                    if verification_result:
                        if "correct" in verification_result.lower() and "incorrect" not in verification_result.lower():
                            correct_count += 1
                else:
                    print(res)
                
        print(correct_count)
        accuracy = correct_count / len(gt_data) * 100

        score = min(max(accuracy - 40, 0),30) / 30 * 100
        return {
            "error": None,
            "score": score,
            "test_set_result": {
                "error": None,
                "message": f"Accuracy: {accuracy}, get {score} points",
                "score": score,
            }
        }
        # return {
        #     "accuracy": accuracy
        # }
