




# 🤖 AgencyBench: Benchmarking the Agentic Intelligence in Real-world Scenarios

<div align="center">

[![Website](https://img.shields.io/badge/🌐_Website-Coming_Soon-blue?style=for-the-badge)](https://github.com/GAIR-NLP/AgencyBench)
[![Paper](https://img.shields.io/badge/📄_Paper-arXiv_Coming_Soon-red?style=for-the-badge)](https://github.com/GAIR-NLP/AgencyBench)
[![License](https://img.shields.io/badge/📜_License-MIT-green?style=for-the-badge)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/GAIR-NLP/AgencyBench?style=for-the-badge&logo=github)](https://github.com/GAIR-NLP/AgencyBench/stargazers)

</div>

AgencyBench is a comprehensive benchmark designed to evaluate the agentic intelligence capabilities of Large Language Models (LLMs). This benchmark tests LLMs across diverse domains and complexity levels, measuring their ability to function as autonomous agents capable of planning, executing, and adapting in complex multi-step scenarios.

## 📰 Recent News

- **[2025/09]** 🎉 AgencyBench is released! 49 challenging subtasks across 10 domains
- **[2025/09]** 📊 Benchmark evaluation framework and baseline results coming soon
- **[2025/09]** 🌐 Official website and leaderboard under development

## 🎯 What is AgencyBench?

AgencyBench evaluates LLMs through **10 distinct tasks** spanning multiple domains including:

- 💻 **Software Engineering**: C++ console applications, Java task management systems
- 🎮 **Game Development**: Advanced AI for strategic games like Gomoku
- ⚙️ **Systems Programming**: Distributed systems, fault tolerance, and self-repair mechanisms
- 🔬 **Research & Analysis**: Dataset discovery, scientific modeling, performance evaluation
- 🧠 **Knowledge Reasoning**: Complex fact-based question answering in sports and finance domains

Each task contains multiple progressive subtasks (**49 total**) that increase in complexity, testing various aspects of agentic behavior such as:

- 🎯 Multi-step reasoning and planning
- 💡 Code generation and system implementation
- 📊 Data analysis and scientific computation
- 🔍 Complex information retrieval and synthesis
- 🤖 Autonomous problem-solving and adaptation

## 🏗️ Project Structure

```
AgencyBench/
├── 📄 README.md                 # Project documentation
├── 📋 bench.txt                 # Original LaTeX specification
├── 📊 category.txt              # Subtask categorization
└── 📁 AgencyBench/              # Task specifications and implementations
    ├── 📁 task1/               # C++ Console Chat System
    │   ├── 📄 Task1_C++_Console_Chat_System.json
    │   └── 📁 workspace/       # C++ implementation workspace
    ├── 📁 task2/               # Java Console Task Manager
    │   ├── 📄 Task2_Java_Console_Task_Manager.json
    │   └── 📁 workspace/       # Java implementation workspace
    ├── 📁 task3/               # Gomoku Battle Game
    │   ├── 📄 Task3_Gomoku_Battle_From_Basics_to_Expert_AI.json
    │   └── 📁 workspace/       # Web game implementation workspace
    ├── 📁 task4/               # Autonomous Self-Repair System
    │   ├── 📄 Task4_From_Deterministic_Event_Generation_to_Autonomous_Self-Repair.json
    │   └── 📁 workspace/       # Python systems implementation workspace
    ├── 📁 task5/               # DynToM Dataset Analysis
    │   ├── 📄 Task5_Comparing_LLM_Performance_on_DynToM_Dataset.json
    │   └── 📁 workspace/       # Research analysis workspace
    ├── 📁 task6/               # GPT-4o Comparative Study
    │   ├── 📄 Task6_Reasoning_vs_Direct_A_Comparative_Study_of_GPT-4o_and_GPT-4o-Reasoning.json
    │   └── 📁 workspace/       # Comparative study workspace
    ├── 📁 task7/               # Dataset Discovery System
    │   ├── 📄 Task7_Three-Stage_Dataset_Discovery_and_Metadata_Extraction.json
    │   └── 📁 workspace/       # Dataset discovery workspace
    ├── 📁 task8/               # Scientific System Function Discovery
    │   ├── 📄 Task8_Scientific_System_Function_Discovery.json
    │   └── 📁 workspace/       # Scientific modeling workspace
    ├── 📁 task9/               # NBA Player Analysis
    │   ├── 📄 Task9_Complex_NBA_Player_Trade_and_Achievement_Scenarios.json
    │   └── 📁 workspace/       # Sports analysis workspace
    └── 📁 task10/              # S&P 500 Companies Analysis
        ├── 📄 Task10_Major_S&P_500_Companies_with_Record_Revenues_and_Leadership.json
        └── 📁 workspace/       # Financial analysis workspace
```

## 🚀 Getting Started

### Quick Start

```bash
git clone https://github.com/GAIR-NLP/AgencyBench.git
cd AgencyBench
```

### Task Format

Each JSON task file contains:
- `metadata`: Task statistics including subtask count and associated capability categories
- `query`: Clean text description of the task requirements

```json
{
  "metadata": {
    "subtask_count": 5,
    "categories": ["User Authentication & Data Persistence", "Social/Graph Feature Implementation", ...]
  },
  "query": "Task description with requirements and success criteria..."
}
```

## 📈 Benchmark Statistics

| Category | Subtasks |
|----------|----------|
| Complex Fact-Based Q&A (Sports/Finance) | 8 |
| Scientific Model/Equation Refinement | 5 |
| Performance Metric Calculation | 4 |
| Dataset Discovery & Metadata Extraction | 3 |
| **Total Categories** | **25** |
| **Total Subtasks** | **49** |



## 📊 Evaluation Metrics

Our evaluation employs **four key metrics** that capture both effectiveness and efficiency dimensions:

- **🎯 First-Turn Functional Completeness (FTFC)**: Measures the percentage of requirements correctly implemented in the initial response, assessing the model's ability to understand and address complex specifications without iteration

- **✅ Success Rate (SR@R)**: Represents the percentage of queries successfully completed within R allocated rounds, indicating overall reliability and robustness across diverse scenarios

- **⚡ Remaining Chances (RC@R)**: Calculates the average number of unused rounds when queries are successfully completed, measuring computational efficiency and resource optimization

- **🔄 Rounds (R)**: Defines the maximum number of interaction rounds allocated for query completion (R=3 in our implementation)

These metrics collectively provide a comprehensive assessment framework that evaluates both the effectiveness of query completion and the efficiency of resource utilization.

## 🏆 Leaderboard

### 🥇 Official Results (R=3)

| **Model** | **FTFC** | **RC** | **SR** |
|-----------|----------|--------|--------|
| 🥇 **anthropic/claude-sonnet-4** | **0.730** | **0.752** | **0.741** |
| 🥈 **gpt-5** | 0.561 | 0.594 | 0.628 |
| 🥉 **GLM 4.5 sft** | 0.717 | 0.742 | 0.746 |
| **GLM 4.5** | 0.378 | 0.500 | 0.474 |
| **qwen/qwen3-235b-a22b-2507** | 0.230 | 0.282 | 0.313 |
| **moonshotai/kimi-k2(0711)** | 0.207 | 0.251 | 0.266 |
| **deepseek/deepseek-chat-v3.1** | 0.106 | 0.119 | 0.133 |

> 🏅 **Claude Sonnet-4** achieves state-of-the-art performance across all metrics, demonstrating superior agentic intelligence capabilities.

*Results are based on comprehensive evaluation across all 10 AgencyBench domains with R=3 rounds maximum.*



## 🔗 Resources

### 🌐 Website
*Coming soon* - Official AgencyBench website with interactive leaderboards and detailed results

### 📄 Paper
*Coming soon* - Comprehensive research paper detailing benchmark design, evaluation methodology, and baseline results

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⭐ Star History

<div align="center">
<i>Star history will be available once the repository is public</i>
</div>

## 🙏 Acknowledgments

We thank the open-source community and all contributors who helped make AgencyBench possible.

## 📖 Citation

If you use AgencyBench in your research, please cite:

```bibtex
@misc{li2025agencybench,
  title={AgencyBench: Benchmarking the Agentic Intelligence in Real-world Scenarios},
  author={Keyu Li and Mohan Jiang and Yang Xiao and Jie Sun and Jifan Lin and Yumin Zhuang and Ji Zeng and Shijie Xia and Qishuo Hua
  and Xuefeng Li and Xiaojie Cai and Dequan Wang and Wenjie Li and Xiang Wang and Pengfei Liu},
  year={2025},
  howpublished={\url{https://github.com/GAIR-NLP/AgencyBench}},
  note={Github Repo}
}
```

---

<div align="center">
Made with ❤️ by the GAIR-NLP Team
</div>
