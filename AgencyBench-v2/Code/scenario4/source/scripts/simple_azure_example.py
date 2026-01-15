import os
import json
from openai import AzureOpenAI
import base64

# 初始化Azure客户端
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION","2025-04-01-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def call_with_image(image_path, problem_text):
    """带图像的API调用"""
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": problem_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ]
    )
    
    return response.choices[0].message.content

def call_without_image(problem_text):
    """不带图像的API调用"""
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
        messages=[
            {"role": "user", "content": problem_text}
        ]
    )
    
    return response.choices[0].message.content

# 使用示例
if __name__ == "__main__":
    # 带图示例
    # result1 = call_with_image("path/to/image.png", "解决这个几何问题")
    # print("带图结果:", result1)
    
    # 不带图示例  
    result2 = call_without_image("判断图的连通性")
    print("不带图结果:", result2) 