import json
import os
from typing import Dict, Any, List, Tuple

def _read_json_file(file_path: str) -> Dict[str, Any]:
    """读取JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _load_ground_truth(task_type: str, references_dir: str) -> Dict[str, Any]:
    """加载ground truth数据"""
    task_dir = os.path.join(references_dir, task_type)
    if not os.path.exists(task_dir):
        return {}
    
    ground_truth = {}
    try:
        # 遍历任务类型目录中的所有子目录
        for item in os.listdir(task_dir):
            item_path = os.path.join(task_dir, item)
            if os.path.isdir(item_path):
                # 每个子目录的名称就是任务ID
                task_id = item
                ex_file = os.path.join(item_path, 'ex.json')
                
                if os.path.exists(ex_file):
                    with open(ex_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'answer' in data:
                            ground_truth[str(task_id)] = data['answer']
                        elif 'label' in data:
                            ground_truth[str(task_id)] = data['label']
                else:
                    # 如果ex.json不存在，则读取example.json
                    example_file = os.path.join(item_path, 'example.json')
                    if os.path.exists(example_file):
                        with open(example_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if 'answer' in data:
                                ground_truth[str(task_id)] = data['answer']
                            elif 'label' in data:
                                ground_truth[str(task_id)] = data['label']
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    
    return ground_truth

def _compare_answers(answer1: Any, answer2: Any) -> bool:
    """比较两个答案是否相等"""
    # print("compare_answers")
    if answer1 is None or answer2 is None:
        return False

    # print(answer1, answer2)
    # 处理布尔值情况
    def normalize_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            value_lower = value.strip().lower()
            if value_lower in ['true', '1', 'yes', 'y']:
                return True
            elif value_lower in ['false', '0', 'no', 'n']:
                return False
        return value
    
    # 先尝试布尔值标准化
    norm1 = normalize_bool(answer1)
    norm2 = normalize_bool(answer2)
    
    # 如果都是布尔值，直接比较
    if isinstance(norm1, bool) and isinstance(norm2, bool):
        return norm1 == norm2
    
    # 如果一个是布尔值，一个不是，尝试转换
    if isinstance(norm1, bool) or isinstance(norm2, bool):
        try:
            # 将非布尔值也尝试转换为布尔值
            if not isinstance(norm1, bool):
                norm1 = normalize_bool(norm1)
            if not isinstance(norm2, bool):
                norm2 = normalize_bool(norm2)
            
            if isinstance(norm1, bool) and isinstance(norm2, bool):
                return norm1 == norm2
        except:
            pass
    
    # 转换为字符串进行比较
    str1 = str(answer1).strip().lower()
    str2 = str(answer2).strip().lower()
    
    # 直接字符串比较
    if str1 == str2:
        return True
    
    # 尝试数值比较
    try:
        float1 = float(str1)
        float2 = float(str2)
        return abs(float1 - float2) < 1e-6
    except ValueError:
        pass
    
    # 处理列表/数组情况
    try:
        # 尝试解析为JSON
        import json
        try:
            json1 = json.loads(str(answer1)) if isinstance(answer1, str) else answer1
            json2 = json.loads(str(answer2)) if isinstance(answer2, str) else answer2
            
            # 如果都是列表，逐个比较
            if isinstance(json1, list) and isinstance(json2, list):
                if len(json1) != len(json2):
                    return False
                for i in range(len(json1)):
                    if not _compare_answers(json1[i], json2[i]):
                        return False
                return True
            
            # 如果都是字典，比较键值对
            if isinstance(json1, dict) and isinstance(json2, dict):
                if json1.keys() != json2.keys():
                    return False
                for key in json1:
                    if not _compare_answers(json1[key], json2[key]):
                        return False
                return True
                
        except:
            pass
    except:
        pass
    
    # 最后的字符串比较
    return str1 == str2

def _calculate_task_stats(task_type: str, outputs_dir: str, ground_truth: Dict[str, Any]) -> Tuple[int, int, int]:
    """计算任务统计信息"""
    task_dir = os.path.join(outputs_dir, task_type)
    
    if not os.path.exists(task_dir):
        return 0, 0, 0
    
    # 总任务数 = ground truth中的任务数
    total_tasks = len(ground_truth)
    
    # 统计有效任务数和成功任务数
    valid_tasks = 0
    successful_tasks = 0
    
    for problem_id in ground_truth:
        result_dir = os.path.join(task_dir, problem_id)
        result_file = os.path.join(result_dir, 'result.json')
        # print(result_file)
        if os.path.exists(result_file):
            valid_tasks += 1
            result = _read_json_file(result_file)
                # print(result)
            # 获取答案
            answer = result.get('answer') 
            if answer is None:
                answer = result.get('label')
            gt_answer = ground_truth[problem_id]
            # print(answer, gt_answer)
            if _compare_answers(answer, gt_answer):
                successful_tasks += 1
    
    return total_tasks, valid_tasks, successful_tasks

def validate_files(outputs_dir: str, references_dir: str, task_types: List[str]) -> Tuple[int, int, int]:
    """严格验证文件结构，确保所有必需的文件都存在"""
    total_files = 0
    valid_files = 0
    
    for task_type in task_types:
        # 检查reference目录是否存在
        ref_task_dir = os.path.join(references_dir, task_type)
        if not os.path.exists(ref_task_dir):
            raise FileNotFoundError(f"Reference directory not found: {ref_task_dir}")
        
        # 检查output目录是否存在
        out_task_dir = os.path.join(outputs_dir, task_type)
        if not os.path.exists(out_task_dir):
            raise FileNotFoundError(f"Output directory not found: {out_task_dir}")
        
        # 遍历reference中的每个任务ID
        for task_id in os.listdir(ref_task_dir):
            ref_task_path = os.path.join(ref_task_dir, task_id)
            if not os.path.isdir(ref_task_path):
                continue
                
            # 检查reference中是否有答案文件
            ex_file = os.path.join(ref_task_path, 'ex.json')
            example_file = os.path.join(ref_task_path, 'example.json')
            
            has_answer = False
            if os.path.exists(ex_file):
                try:
                    with open(ex_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'answer' in data or 'label' in data:
                            has_answer = True
                except (json.JSONDecodeError, IOError):
                    pass
            
            if not has_answer and os.path.exists(example_file):
                try:
                    with open(example_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'answer' in data or 'label' in data:
                            has_answer = True
                except (json.JSONDecodeError, IOError):
                    pass
            
            if not has_answer:
                raise ValueError(f"No answer found in reference for task {task_type}/{task_id}")
            
            # 检查output中对应的result.json文件
            out_task_path = os.path.join(out_task_dir, task_id)
            if not os.path.exists(out_task_path):
                raise FileNotFoundError(f"Output task directory not found: {out_task_path}")
            
            result_file = os.path.join(out_task_path, 'result.json')
            if not os.path.exists(result_file):
                raise FileNotFoundError(f"Result file not found: {result_file}")
            
            total_files += 1
            
            # 检查result.json是否包含答案
            try:
                result = _read_json_file(result_file)
                if 'answer' in result or 'label' in result:
                    valid_files += 1
                else:
                    raise ValueError(f"No answer/label found in result file: {result_file}")
            except Exception as e:
                raise ValueError(f"Error reading result file {result_file}: {str(e)}")
    
    # 计算文件验证得分
    score = 0
    if total_files > 0:
        score = int((valid_files / total_files) * 10)
    
    return total_files, valid_files, score

def calculate_task_score(outputs_dir: str, references_dir: str, task_types: List[str]) -> Tuple[int, int, int, Dict[str, Any]]:
    """计算任务得分"""
    total_tasks_all = 0
    total_successful_all = 0
    task_details = {}
    
    for task_type in task_types:
        ground_truth = _load_ground_truth(task_type, references_dir)
        # print(ground_truth)
        if not ground_truth:
            task_details[f"{task_type}_success_rate"] = 0.0
            task_details[f"{task_type}_total_tasks"] = 0
            task_details[f"{task_type}_successful_tasks"] = 0
            continue
        
        total_tasks, valid_tasks, successful_tasks = _calculate_task_stats(task_type, outputs_dir, ground_truth)
        
        # 累加到总数
        total_tasks_all += total_tasks
        total_successful_all += successful_tasks
        
        # 计算单个任务类型的成功率
        success_rate = 0.0
        if total_tasks > 0:
            success_rate = (successful_tasks / total_tasks) * 100
        
        task_details[f"{task_type}_success_rate"] = success_rate
        task_details[f"{task_type}_total_tasks"] = total_tasks
        task_details[f"{task_type}_successful_tasks"] = successful_tasks
    
    # 计算总体任务得分 (总成功任务数 / 总任务数) × 90
    task_score = 0
    if total_tasks_all > 0:
        task_score = max(min(total_successful_all / total_tasks_all * 90, 90), 0)
    
    return total_tasks_all, total_successful_all, task_score, task_details

# 保留原有函数以保持兼容性
def calculate_overall_score(outputs_dir: str, references_dir: str, task_types: List[str]) -> Tuple[float, int, int]:
    """计算整体成功率（保留兼容性）"""
    total_valid = 0
    total_successful = 0
    
    for task_type in task_types:
        ground_truth = _load_ground_truth(task_type, references_dir)
        if not ground_truth:
            continue
        
        _, valid_tasks, successful_tasks = _calculate_task_stats(task_type, outputs_dir, ground_truth)
        total_valid += valid_tasks
        total_successful += successful_tasks
    
    success_rate = 0.0
    if total_valid > 0:
        success_rate = (total_successful / total_valid) * 100
    
    score = int(success_rate)
    return success_rate, score, total_valid

def calculate_detailed_scores(outputs_dir: str, references_dir: str, task_types: List[str]) -> Dict[str, Any]:
    """计算详细的任务类型分数（保留兼容性）"""
    results = {}
    total_score = 0
    
    for task_type in task_types:
        ground_truth = _load_ground_truth(task_type, references_dir)
        if not ground_truth:
            results[f"{task_type}_success_rate"] = 0.0
            results[f"{task_type}_score"] = 0
            results[f"{task_type}_valid_tasks"] = 0
            continue
        
        total_tasks, valid_tasks, successful_tasks = _calculate_task_stats(task_type, outputs_dir, ground_truth)
        
        success_rate = 0.0
        if valid_tasks > 0:
            success_rate = (successful_tasks / valid_tasks) * 100
        
        score = max(min(int(success_rate / len(task_types) * 10), 10), 0)
        total_score += score
        
        results[f"{task_type}_success_rate"] = success_rate
        results[f"{task_type}_score"] = score
        results[f"{task_type}_valid_tasks"] = valid_tasks
    
    results["detailed_score"] = total_score
    return results