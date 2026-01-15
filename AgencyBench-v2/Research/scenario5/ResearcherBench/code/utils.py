#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for AI model response evaluation
"""

import os
import re
import json
from typing import List, Dict, Any

def load_json_file(file_path: str) -> Any:
    """
    Load JSON file with error handling
    
    Parameters:
    file_path (str): Path to the JSON file
    
    Returns:
    Any: Loaded JSON data or None if error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded: {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {file_path} - {e}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def save_json_file(data: Any, file_path: str) -> bool:
    """
    Save data to JSON file with error handling
    
    Parameters:
    data (Any): Data to save
    file_path (str): Path to save the file
    
    Returns:
    bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved: {file_path}")
        return True
    except Exception as e:
        print(f"Error saving {file_path}: {e}")
        return False


def load_rubrics(rubrics_file: str) -> Dict[int, Dict]:
    """
    Load rubrics from JSON file and organize by question ID
    
    Parameters:
    rubrics_file (str): Path to rubrics JSON file
    
    Returns:
    Dict[int, Dict]: Dictionary mapping question ID to rubric data
    """
    rubrics_data = load_json_file(rubrics_file)
    if not rubrics_data:
        return {}
    
    rubrics_dict = {}
    for item in rubrics_data:
        question_id = item.get("id")
        if question_id:
            rubrics_dict[question_id] = item
    
    print(f"Loaded rubrics for {len(rubrics_dict)} questions")
    return rubrics_dict


def load_model_responses(responses_file: str) -> Dict[int, Dict]:
    """
    Load model responses from JSON file and organize by question ID
    
    Parameters:
    responses_file (str): Path to model responses JSON file
    
    Returns:
    Dict[int, Dict]: Dictionary mapping question ID to response data
    """
    responses_data = load_json_file(responses_file)
    if not responses_data:
        return {}
    
    responses_dict = {}
    for item in responses_data:
        question_id = item.get("id")
        if question_id:
            responses_dict[question_id] = item
    
    print(f"Loaded responses for {len(responses_dict)} questions")
    return responses_dict


def get_model_name_from_file(file_path: str) -> str:
    """
    Extract model name from file path
    
    Parameters:
    file_path (str): Full path to the model response file
    
    Returns:
    str: Model name extracted from filename
    """
    filename = os.path.basename(file_path)
    # Remove .json extension
    model_name = filename.replace('.json', '')
    return model_name


def create_output_directory(output_dir: str) -> bool:
    """
    Create output directory if it doesn't exist
    
    Parameters:
    output_dir (str): Directory path to create
    
    Returns:
    bool: True if successful, False otherwise
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {output_dir}: {e}")
        return False


def validate_rubric_format(rubric_data: Dict) -> bool:
    """
    Validate rubric data format
    
    Parameters:
    rubric_data (Dict): Rubric data to validate
    
    Returns:
    bool: True if valid format, False otherwise
    """
    required_fields = ["id", "question", "rubric"]
    
    for field in required_fields:
        if field not in rubric_data:
            print(f"Missing required field: {field}")
            return False
    
    if not isinstance(rubric_data["rubric"], list):
        print("Rubric should be a list")
        return False
    
    for item in rubric_data["rubric"]:
        if not isinstance(item, dict) or "point" not in item or "weight" not in item:
            print("Invalid rubric item format")
            return False
    
    return True


def validate_response_format(response_data: Dict) -> bool:
    """
    Validate response data format
    
    Parameters:
    response_data (Dict): Response data to validate
    
    Returns:
    bool: True if valid format, False otherwise
    """
    required_fields = ["id", "question", "response"]
    
    for field in required_fields:
        if field not in response_data:
            print(f"Missing required field: {field}")
            return False
    
    return True


def get_api_config(model_name: str) -> Dict[str, str]:
    """
    Get API configuration for different models
    
    Parameters:
    model_name (str): Name of the model to use for evaluation
    
    Returns:
    Dict[str, str]: API configuration dictionary
    """
    configs = {
        "o3-mini": {
            "model": "o3-mini"
        },
        "gpt-4o": {
            "model": "gpt-4o"
        }
    }
    
    return configs.get(model_name, configs["o3-mini"])  # Default to o3-mini

def fix_json_quotes(json_str):
    """修复JSON字符串中的引号问题"""
    def fix_quotes_in_strings(match):
        key_part = match.group(1)  # "key": 
        value_content = match.group(2)  # 引号内的内容
        # 转义内容中的双引号
        fixed_content = value_content.replace('"', '\\"')
        return f'{key_part}"{fixed_content}"'
    
    # 匹配 "key": "value" 格式，其中value可能包含未转义的引号
    pattern = r'("[\w\s]+"\s*:\s*)"([^"]*(?:"[^"]*)*)"'
    fixed_str = re.sub(pattern, fix_quotes_in_strings, json_str)
    
    return fixed_str

def clean_invalid_escapes(content):
    """清理JSON中的无效转义字符"""
    # 先处理双重转义的情况
    content = content.replace("\\\\", "\\")
    
    # 定义所有需要清理的无效转义字符
    invalid_escapes = [
        ("\\(", "("), ("\\)", ")"),
        ("\\[", "["), ("\\]", "]"),
        ("\\{", "{"), ("\\}", "}"),
        ("\\>", ">"), ("\\<", "<"),
        ("\\+", "+"), ("\\-", "-"),
        ("\\*", "*"), ("\\=", "="),
        ("\\~", "~"), ("\\@", "@"),
        ("\\#", "#"), ("\\%", "%"),
        ("\\&", "&"), ("\\|", "|"),
        ("\\^", "^"), ("\\$", "$"),
        ("\\!", "!"), ("\\?", "?"),
        ("\\:", ":"), ("\\;", ";"),
        ("\\.", "."), ("\\,", ","),
        ("\\log", "log"), ("\\sin", "sin"), ("\\cos", "cos"),
        ("\\_", "_"), ("\\space", " "),
    ]
    
    # 批量替换无效转义
    for invalid, valid in invalid_escapes:
        content = content.replace(invalid, valid)
    
    # 处理数字后面的反斜杠（如 2211.09110\]）
    # 匹配数字后跟反斜杠和特殊字符的模式
    content = re.sub(r'(\d+\.?\d*)\\\]', r'\1]', content)
    content = re.sub(r'(\d+\.?\d*)\\\[', r'\1[', content)
    content = re.sub(r'(\d+\.?\d*)\\\)', r'\1)', content)
    content = re.sub(r'(\d+\.?\d*)\\\(', r'\1(', content)
    
    # 处理其他常见的无效转义模式
    # 匹配反斜杠后跟非标准转义字符的情况
    content = re.sub(r'\\([^\\"\/bfnrtuxU])', r'\1', content)
    
    return content

def json_safe_loads(content):
    """安全的JSON加载函数"""
    # 第一步：清理无效转义字符
    content = clean_invalid_escapes(content)
    
    try:
        # 首先尝试直接解析
        return json.loads(content)
    except json.JSONDecodeError as e:
        try:
            # 尝试修复引号后再解析
            fixed_content = fix_json_quotes(content)
            return json.loads(fixed_content)
        except json.JSONDecodeError as e2:
            try:
                # 再次清理可能在引号修复过程中产生的问题
                double_fixed = clean_invalid_escapes(fixed_content)
                return json.loads(double_fixed)
            except json.JSONDecodeError as e3:
                print(f"Failed to parse JSON content: {str(e3)}")
                try:
                    # 生成一段乱码作为id
                    random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
                    with open(f"error_file/error_claims_{random_id}.json", "w", encoding="utf-8") as f:
                        f.write(f"Content:\n{double_fixed}\n\n")
                        f.write(f"Error: \n{str(e3)}")
                    print(f"Invalid JSON saved to error_file/error_claims_{random_id}.json")
                except Exception as file_error:
                    print(f"Error saving invalid JSON: {str(file_error)}")
                return []