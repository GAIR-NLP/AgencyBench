




# 🤖 AgencyBench: Benchmarking the Agentic Intelligence of LLMs

<div align="center">

[![Website](https://img.shields.io/badge/🌐_Website-Coming_Soon-blue?style=for-the-badge)](https://github.com/GAIR-NLP/AgencyBench)
[![Paper](https://img.shields.io/badge/📄_Paper-arXiv_Coming_Soon-red?style=for-the-badge)](https://github.com/GAIR-NLP/AgencyBench)
[![License](https://img.shields.io/badge/📜_License-MIT-green?style=for-the-badge)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/GAIR-NLP/AgencyBench?style=for-the-badge&logo=github)](https://github.com/GAIR-NLP/AgencyBench/stargazers)

</div>

AgencyBench is a comprehensive benchmark designed to evaluate the agentic intelligence capabilities of Large Language Models (LLMs). This benchmark tests LLMs across diverse domains and complexity levels, measuring their ability to function as autonomous agents capable of planning, executing, and adapting in complex multi-step scenarios.

## 📰 Recent News

- **[2025/01]** 🎉 AgencyBench is released! 49 challenging subtasks across 10 domains
- **[2025/01]** 📊 Benchmark evaluation framework and baseline results coming soon
- **[2025/01]** 🌐 Official website and leaderboard under development

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
└── 📁 AgencyBench/              # Task specifications (JSON format)
    ├── Task1_C++_Console_Chat_System.json
    ├── Task2_Java_Console_Task_Manager.json
    ├── Task3_Gomoku_Battle_From_Basics_to_Expert_AI.json
    ├── Task4_From_Deterministic_Event_Generation_to_Autonomous_Self-Repair.json
    ├── Task5_Comparing_LLM_Performance_on_DynToM_Dataset.json
    ├── Task6_Reasoning_vs_Direct_A_Comparative_Study_of_GPT-4o_and_GPT-4o-Reasoning.json
    ├── Task7_Three-Stage_Dataset_Discovery_and_Metadata_Extraction.json
    ├── Task8_Scientific_System_Function_Discovery.json
    ├── Task9_Complex_NBA_Player_Trade_and_Achievement_Scenarios.json
    └── Task10_Major_S&P_500_Companies_with_Record_Revenues_and_Leadership.json
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

## 🔗 Resources

### 🌐 Website
*Coming soon* - Official AgencyBench website with interactive leaderboards and detailed results

### 📄 Paper
*Coming soon* - Comprehensive research paper detailing benchmark design, evaluation methodology, and baseline results

## 🤝 Contributing

We welcome contributions to AgencyBench! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to:

- 📝 Submit new tasks or improvements
- 🐛 Report bugs and issues
- 💡 Suggest new features
- 📋 Add evaluation metrics

## 🏆 Leaderboard

*Coming soon* - Official leaderboard with model performance rankings across all tasks

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GAIR-NLP/AgencyBench&type=Date)](https://star-history.com/#GAIR-NLP/AgencyBench&Date)

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
