# Agent MVP 配置说明

## 环境变量配置

编辑 `.env` 文件，设置以下变量：

### 必需配置
```bash
# GitHub Token (必需)
GITHUB_TOKEN=ghp_your_github_token_here

# LLM API 配置 (必需)
LLM_API_KEY=your_llm_api_key_here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

# Webhook 配置
WEBHOOK_SECRET=your_webhook_secret_here
WEBHOOK_PORT=18234

# MongoDB 配置 (已配置)
MONGODB_URL=mongodb://ghagent:Ghagent2025@mongoreplicaed6a95949b2c0.mongodb.cn-hongkong.ivolces.com:3717,mongoreplicaed6a95949b2c1.mongodb.cn-hongkong.ivolces.com:3717/?authSource=admin&replicaSet=rs-mongo-replica-ed6a95949b2c&retryWrites=true
MONGODB_DB=gh_agent_mvp

# 并发限制
MAX_CONCURRENT_AGENTS=2
```

## 支持的 LLM 模型

### OpenAI 兼容 API
- `gpt-4o` - GPT-4 Omni (推荐)
- `gpt-4o-mini` - GPT-4 Omni Mini (默认，性价比高)
- `gpt-4-turbo` - GPT-4 Turbo
- `gpt-3.5-turbo` - GPT-3.5 Turbo

### 其他兼容 API
- `claude-3-5-sonnet-20241022` - Anthropic Claude
- `gemini-pro` - Google Gemini
- `llama-3.1-8b` - Meta Llama

## GitHub Webhook 配置

1. 进入 GitHub 仓库 → Settings → Webhooks
2. 点击 "Add webhook"
3. 配置：
   - **Payload URL**: `http://your-server-ip:18234/webhooks/github`
   - **Content type**: `application/json`
   - **Secret**: 与 `.env` 中的 `WEBHOOK_SECRET` 一致
   - **Events**: 选择 `Issues` 和 `Issue comments`
   - **Active**: ✅

## 启动服务

```bash
cd /home/gh_agent/gh_agent
./start.sh
```

## 测试

### 健康检查
```bash
curl http://localhost:18234/health
```

### 测试 Webhook
```bash
# Ping 测试
curl -X POST http://localhost:18234/webhooks/github \
  -H "X-GitHub-Event: ping" \
  -H "X-Hub-Signature-256: sha256=5ad1ed89ef07c9d0aa87317619ced9f3b45ab82d99e66145d89bca3e4bf94e12" \
  -d '{"test": "ping"}'

# Issues 测试
curl -X POST http://localhost:18234/webhooks/github \
  -H "X-GitHub-Event: issues" \
  -H "X-Hub-Signature-256: sha256=87f9a75f3c54340b031b8bfdd82b30c27e22b86f9c38f7e5ec2219efeddf65a3" \
  -H "Content-Type: application/json" \
  -d '{"action": "assigned", "issue": {"number": 123, "title": "Test Issue", "body": "This is a test issue for the agent"}, "repository": {"full_name": "test/repo"}}'
```

## 监控

- **健康检查**: `http://localhost:18234/health`
- **API 文档**: `http://localhost:18234/docs`
- **监控页面**: `http://localhost:18234/static/monitor.html`

## 故障排除

1. **服务启动失败**: 检查端口是否被占用
2. **Webhook 验证失败**: 检查 `WEBHOOK_SECRET` 配置
3. **LLM 调用失败**: 检查 `LLM_API_KEY` 和 `LLM_BASE_URL`
4. **数据库连接失败**: 检查 `MONGODB_URL` 配置
