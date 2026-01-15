# MVP 版本方案

## 技术栈选型

### 核心组件
- **GitHub CLI**: 认证和代码库管理，替代 GitHub API 客户端
- **GitHub Webhook**: 监听 `issues.assigned` 和 `issue_comment.created` 事件
- **SmolAgents**: [Code Agent Core](https://github.com/huggingface/smolagents) 作为 Agent 执行引擎
- **本地沙箱**: Docker 容器或 E2B 本地部署提供隔离的代码执行环境
- **MongoDB**: 火山云服务，存储 Agent 会话和日志数据

### LLM 服务
- **OpenAI Compatible API**: 支持多个固定模型调用
- 模型选择策略：根据任务复杂度自动选择合适模型

### 前端技术
- **Tailwind CSS + Shadcn/ui**: 构建监控网页界面
- **实时数据更新**: WebSocket 或 Server-Sent Events

### 可扩展组件
- **BrowserUse**: 自动化测试（暂无推荐方案，待集成）

## Webhook 监听配置
基于 [GitHub Webhook](https://docs.github.com/zh/webhooks/about-webhooks) 监听以下事件：
- `issues.assigned`: Issue 分配给 Agent 用户时触发
- `issue_comment.created`: 用户 @ 到 Agent 用户时触发

**Webhook 配置要求：**
- 监听仓库：目标开发仓库（但不一定只有一个仓库）
- 内容类型：`application/json`
- 密钥：用于验证 webhook 请求的合法性
- URL：本地服务地址（需要公网可访问，这个本地已经在云服务器上）

## Agent 工作流程
1. **接收 Webhook 事件** → 验证签名 → 解析 Issue 信息
2. **回复确认** → 使用 GitHub CLI 在 Issue 中回复"收到任务，开始处理"
3. **创建沙箱** → 使用 Docker 容器或 E2B 本地部署创建隔离环境并克隆目标仓库
4. **理解任务** → 使用 SmolAgents + LLM 分析 Issue 内容，梳理涉及文件
5. **执行开发** → 在 Sandbox 中编写代码实现需求
6. **验证结果** → 通过以下方式之一验证：
   - BrowserUse 自动化测试（可扩展）
   - 接口测试
   - 提供部署文档供人工测试
7. **提交结果** → 使用 GitHub CLI 创建分支 → 提交 PR → 附带测试报告

**失败处理：**
- 用户给任务就尝试，失败直接报告
- 过大 Issue 导致失败属于正常情况，无需特殊处理
- 失败后在 Issue 中回复状态，可选择提交失败状态的 PR
- Agent 退出时自动清理对应的沙箱环境

**挂起机制：**
- 当 Agent 需要用户输入时，状态变为 `suspended`
- 保持沙箱环境不清理，等待用户响应
- 用户通过监控界面或 Issue 评论提供信息后，Agent 恢复执行

## 监控网页 (Tailwind + Shadcn)
**展示内容：**
- Agent 运行状态（运行中/已结束）
- 实时日志（从 MongoDB 读取）
- 运行历史记录
- 关联的 Issue 和 PR 列表

**交互功能：**
- 查看实时日志流
- 手动停止正在运行的 Agent（发送 CancelException）
- 用户输入交互

## 技术架构

### 后端服务架构
- **FastAPI 服务**：接收 GitHub Webhook，提供 REST API
- **SmolAgents 引擎**：执行 Agent 任务，与 LLM 交互
- **沙箱管理器**：管理 Docker 容器或 E2B 本地环境创建和销毁
- **GitHub CLI 集成**：通过 subprocess 调用 GitHub CLI 命令
- **MongoDB 客户端**：存储会话数据和日志
- **WebSocket 服务**：实时推送日志到前端

### 数据库 (MongoDB)
存储结构：

**agent_sessions 集合：**
```json
{
  "session_id": "uuid",
  "issue_number": 123,
  "repository": "owner/repo",
  "status": "running|completed|failed|suspended",
  "start_time": "2024-01-01T00:00:00Z",
  "end_time": "2024-01-01T01:00:00Z",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z",
  "sandbox_id": "container_id_or_sandbox_id"
}
```

**agent_logs 集合：**
```json
{
  "session_id": "uuid",
  "timestamp": "2024-01-01T00:00:00Z",
  "level": "info|debug|error|warning",
  "message": "log message content",
  "source": "agent|webhook|github_api"
}
```

**索引设计：**
- `agent_sessions.session_id` (唯一索引)
- `agent_sessions.repository + issue_number` (复合唯一索引，防止重复处理)
- `agent_sessions.status` (普通索引，用于查询运行状态)
- `agent_logs.session_id` (普通索引)
- `agent_logs.timestamp` (普通索引)

### 运行限制
- **并发数**：最多 2 个 Agent 同时运行
- **超时时间**：每个 Agent 最多运行 1 小时
- **权限范围**：Write 权限，只能创建分支和 PR，不能直接合并到 main
- **重复处理防护**：同一时间只有一个 Agent 处理同一个 Issue
- **沙箱清理**：Agent 完成或失败后自动清理对应的沙箱环境
- **挂起机制**：Agent 等待用户输入时进入 suspended 状态，保持沙箱环境

### 环境配置
```bash
# .env 文件
GITHUB_TOKEN=your_github_token
LLM_API_KEY=your_llm_api_key
LLM_BASE_URL=https://api.openai.com/v1  # 或其他兼容 API
MONGODB_URL=your_mongodb_url
WEBHOOK_SECRET=your_webhook_secret
# E2B_API_KEY=your_e2b_api_key  # 如果使用 E2B 本地部署
```

### 依赖包 (requirements.txt)
```
fastapi
uvicorn
pymongo
smolagents
docker
python-dotenv
websockets
```

### 部署方式
- **本地部署**：Python 服务 + 监控网页
- **公网访问**：使用 ngrok 或云服务器提供 webhook URL
- **数据库**：云端 MongoDB，无需本地备份

### 代码管理
- 本地 Git 仓库管理 Agent 代码
- 暂时不链接到远程仓库