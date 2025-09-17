




# AgencyBench: Benchmarking the Agentic Intelligence of LLMs

AgencyBench is a comprehensive benchmark designed to evaluate the agentic intelligence capabilities of Large Language Models (LLMs). This benchmark tests LLMs across diverse domains and complexity levels, measuring their ability to function as autonomous agents capable of planning, executing, and adapting in complex multi-step scenarios.

## What is AgencyBench?

AgencyBench evaluates LLMs through 10 distinct tasks spanning multiple domains including:
- **Software Engineering**: C++ console applications, Java task management systems
- **Game Development**: Advanced AI for strategic games like Gomoku
- **Systems Programming**: Distributed systems, fault tolerance, and self-repair mechanisms
- **Research & Analysis**: Dataset discovery, scientific modeling, performance evaluation
- **Knowledge Reasoning**: Complex fact-based question answering in sports and finance domains

Each task contains multiple progressive subtasks (49 total) that increase in complexity, testing various aspects of agentic behavior such as:
- Multi-step reasoning and planning
- Code generation and system implementation
- Data analysis and scientific computation
- Complex information retrieval and synthesis
- Autonomous problem-solving and adaptation

## Project Structure

```
AgencyBench/
├── README.md                     # This file
├── bench.txt                     # Original LaTeX specification of all tasks
├── category.txt                  # Categorization table of all subtasks
├── parse_tasks.py               # Script to convert LaTeX tasks to JSON format
└── AgencyBench/                 # Task specifications in JSON format
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

### File Descriptions

- **bench.txt**: Contains the complete LaTeX specification of all 10 tasks with detailed requirements and subtasks
- **category.txt**: Statistical breakdown of the 49 subtasks organized by 25 different capability categories
- **parse_tasks.py**: Python utility script that converts LaTeX task descriptions into structured JSON format
- **AgencyBench/*.json**: Individual task files containing structured metadata (subtask count, categories) and cleaned task descriptions

Each JSON task file includes:
- `metadata`: Task statistics including subtask count and associated capability categories
- `query`: Clean text version of the task description with LaTeX formatting removed

## Resources

### Website
*Coming soon* - Official AgencyBench website with interactive leaderboards and detailed results

### Paper
*Coming soon* - Comprehensive research paper detailing benchmark design, evaluation methodology, and baseline results

## Citation

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
