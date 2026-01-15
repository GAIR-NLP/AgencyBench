# Visual Sketchpad 实现指导

## 核心框架概述

Visual Sketchpad是一个使多模态大语言模型能够生成中间视觉草图作为推理步骤的框架。核心思想是实现**思考-行动-观察**循环，模型可以绘制视觉工件来促进数学推理。

### 关键组件

1. **多模态推理引擎**：处理文本和视觉输入
2. **代码生成模块**：生成Python代码来创建视觉草图
3. **可视化工具**：如matplotlib、networkx、chess等用于不同任务的库
4. **执行环境**：运行生成的代码并捕获视觉输出

## 实现策略

### 1. 框架架构

```python
class VisualSketchpad:
    def __init__(self, model_api):
        self.model = model_api
        self.conversation_history = []
        self.visual_artifacts = []
    
    def solve_problem(self, problem_data, task_type):
        # 实现主要推理循环
        while not self.is_solved():
            thought = self.generate_thought()
            action = self.generate_action(thought)
            observation = self.execute_action(action)
            self.update_context(thought, action, observation)
        
        return self.get_final_answer()
```

### 2. 思考-行动-观察循环

**思考阶段**：
- 分析当前问题状态
- 规划下一个要生成的视觉草图
- 决定需要什么信息

**行动阶段**：
- 生成Python代码来创建可视化
- 执行代码产生视觉工件
- 处理错误和调试

**观察阶段**：
- 分析生成的视觉工件
- 从可视化中提取洞察
- 更新推理上下文

### 3. 任务特定实现

#### 几何问题
```python
def solve_geometry(self, problem):
    # 解析几何图形
    # 生成matplotlib代码来绘制辅助线
    # 分析角度和关系
    
    action_code = '''
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # 绘制原始图形
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # 根据推理添加辅助线
    # 例如：绘制平行线、垂直线等
    
    plt.savefig('geometry_sketch.png', dpi=150, bbox_inches='tight')
    plt.show()
    '''
    
    return self.execute_and_observe(action_code)
```

#### 图论问题（连通性、同构、最大流）
```python
def solve_graph_problem(self, adjacency_matrix, task_type):
    action_code = '''
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # 从邻接矩阵创建图
    G = nx.from_numpy_array(adjacency_matrix)
    
    # 不同任务使用不同布局
    if task_type == "connectivity":
        pos = nx.spring_layout(G)
        # 高亮连通组件
        
    elif task_type == "maxflow":
        pos = nx.spring_layout(G)
        # 绘制边权重和流量值
        
    elif task_type == "isomorphism":
        # 并排绘制两个图进行比较
        pass
    
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=16, font_weight='bold')
    
    plt.savefig('graph_sketch.png', dpi=150, bbox_inches='tight')
    plt.show()
    '''
    
    return self.execute_and_observe(action_code)
```

#### 数学函数问题（奇偶性、凸性）
```python
def solve_function_problem(self, function_expr, task_type):
    action_code = f'''
    import numpy as np
    import matplotlib.pyplot as plt
    import sympy as sp
    
    # 解析函数表达式
    x = sp.Symbol('x')
    f = sp.parse_expr("{function_expr}")
    
    # 创建数值函数用于绘图
    f_lambda = sp.lambdify(x, f, 'numpy')
    
    # 生成x值
    x_vals = np.linspace(-5, 5, 1000)
    y_vals = f_lambda(x_vals)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'f(x) = {function_expr}')
    
    if task_type == "parity":
        # 同时绘制-f(-x)来检查奇偶对称性
        plt.plot(x_vals, -f_lambda(-x_vals), 'r--', linewidth=2, label='-f(-x)')
        plt.plot(x_vals, f_lambda(-x_vals), 'g--', linewidth=2, label='f(-x)')
    
    elif task_type == "convexity":
        # 添加二阶导数可视化
        f_prime = sp.diff(f, x)
        f_double_prime = sp.diff(f_prime, x)
        f_dp_lambda = sp.lambdify(x, f_double_prime, 'numpy')
        
        plt.subplot(2, 1, 1)
        plt.plot(x_vals, y_vals, 'b-', linewidth=2)
        plt.title('函数')
        
        plt.subplot(2, 1, 2)
        plt.plot(x_vals, f_dp_lambda(x_vals), 'r-', linewidth=2)
        plt.title('二阶导数')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('function_sketch.png', dpi=150, bbox_inches='tight')
    plt.show()
    '''
    
    return self.execute_and_observe(action_code)
```

#### 国际象棋胜负识别
```python
def solve_chess_problem(self, fen_string):
    action_code = f'''
    import chess
    import chess.svg
    import matplotlib.pyplot as plt
    from PIL import Image
    import cairosvg
    import io
    
    # 解析FEN字符串
    board = chess.Board("{fen_string}")
    
    # 生成棋盘可视化
    svg_data = chess.svg.board(board, size=400)
    
    # 转换SVG为PNG
    png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
    image = Image.open(io.BytesIO(png_data))
    
    # 显示和保存
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'象棋位置\\nFEN: {fen_string}')
    plt.savefig('chess_sketch.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 同时分析位置
    print(f"物质平衡: {chess.PIECE_VALUES}")
    print(f"是否将死: {board.is_checkmate()}")
    print(f"是否和棋: {board.is_stalemate()}")
    '''
    
    return self.execute_and_observe(action_code)
```

### 4. API集成

```python
import openai
import os

def setup_openai_client():
    # 首先尝试Azure OpenAI
    if os.getenv('AZURE_OPENAI_API_KEY'):
        return openai.AzureOpenAI(
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            api_version="2024-02-01",
            azure_endpoint=os.getenv('AZURE_OPENAI_BASE_URL')
        )
    else:
        # 回退到OpenAI
        return openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL')
        )

def generate_reasoning_step(client, context, problem_data):
    messages = [
        {"role": "system", "content": "你是一个使用视觉草图解决问题的数学推理助手。"},
        {"role": "user", "content": f"问题: {problem_data}\n上下文: {context}\n生成你的下一个推理步骤。"}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1000,
        temperature=0.1
    )
    
    return response.choices[0].message.content
```

### 5. 错误处理和健壮性

```python
def safe_execute_code(self, code):
    try:
        # 在受控环境中执行
        exec_globals = {
            'plt': plt,
            'np': np,
            'nx': nx,
            'sp': sp,
            'chess': chess,
            '__builtins__': __builtins__
        }
        
        exec(code, exec_globals)
        return True, "代码执行成功"
    
    except Exception as e:
        # 尝试修复常见问题
        if "division by zero" in str(e):
            # 处理数学错误
            pass
        elif "import" in str(e):
            # 处理缺失导入
            pass
        
        return False, f"执行错误: {str(e)}"
```

### 6. 多步推理模式

```python
def solve_with_visual_reasoning(self, problem_data, task_type):
    reasoning_steps = []
    
    # 步骤1：初始分析
    thought1 = self.generate_thought("分析问题结构")
    action1 = self.generate_visualization_code(thought1, problem_data)
    observation1 = self.execute_and_observe(action1)
    
    # 步骤2：详细推理
    thought2 = self.generate_thought("基于可视化，我们能获得什么洞察？")
    action2 = self.generate_analysis_code(thought2, observation1)
    observation2 = self.execute_and_observe(action2)
    
    # 步骤3：最终答案
    final_answer = self.synthesize_answer(reasoning_steps)
    
    return {
        "answer": final_answer,
        "reasoning_steps": reasoning_steps,
        "visual_artifacts": self.visual_artifacts
    }
```

### 7. 性能优化技巧

1. **高效代码生成**：为常见可视化模式使用模板
2. **缓存**：缓存频繁使用的计算和可视化
3. **并行处理**：为复杂问题运行多个推理线程
4. **智能提示**：使用少样本示例来指导模型

### 8. 输出格式

```python
def save_results(self, task_type, problem_id, results):
    output_dir = f"gym_workspace/data/task_0021_1/outputs/{task_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    result_file = os.path.join(output_dir, problem_id, "result.json")
    
    with open(result_file, 'w') as f:
        json.dump({
            "problem_id": problem_id,
            "task_type": task_type,
            "answer": results["answer"],
            "reasoning_steps": results["reasoning_steps"],
            "visual_artifacts": results["visual_artifacts"],
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
```

### 9. 集成示例

```python
def main():
    # 初始化Visual Sketchpad系统
    client = setup_openai_client()
    sketchpad = VisualSketchpad(client)
    
    # 加载所有数据集
    datasets = load_all_datasets()
    
    # 处理每个任务类型
    for task_type in ["geometry", "graph_connectivity", "graph_isomorphism", 
                      "graph_maxflow", "math_convexity", "math_parity", "winner_id"]:
        
        print(f"处理 {task_type} 任务...")
        
        for problem_id, problem_data in datasets[task_type].items():
            try:
                results = sketchpad.solve_problem(problem_data, task_type)
                save_results(task_type, problem_id, results)
                
            except Exception as e:
                print(f"处理 {task_type}/{problem_id} 时出错: {str(e)}")
                
    print("所有任务完成！")

if __name__ == "__main__":
    main()
```

## 高级技术

### 1. 多模态提示工程
- 结合文本描述和视觉提示
- 使用结构化提示进行一致的推理
- 实现带有视觉步骤的思维链

### 2. 迭代细化
- 允许模型细化其视觉草图
- 实现错误纠正的反馈循环
- 对复杂问题使用多种可视化方法

### 3. 性能监控
- 跟踪不同问题类型的准确性
- 监控视觉草图质量和相关性
- 实现自动质量评估

这个框架为实现数学推理任务的Visual Sketchpad提供了坚实的基础。关键是将可视化策略适应每个特定的问题类型，同时在所有任务中保持一致的推理模式。 