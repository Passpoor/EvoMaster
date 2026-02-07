# Calculation path adaptor (bohr-agent-sdk)

与 _tmp/MatMaster 一致：**HTTPS 存储走 Bohrium 鉴权**（storage type https + Bohrium plugin），executor 为 None，供 **calculation** 类 MCP 工具使用。

## 注入参数

- **executor**: 始终传 `None`。
- **storage**: `{"type": "https", "plugin": {"type": "bohrium", "access_key": ..., "project_id": ..., "app_key": "agent"}}`，从环境变量读取：
  - `BOHRIUM_ACCESS_KEY`、`BOHRIUM_PROJECT_ID`（建议放在 `.env`）。
- **输入路径**：本地/workspace 文件在配置了 OSS 时上传并替换为 https URL 再调用 MCP。

## /workspace 映射

Agent 可能传入 `/workspace/Fe_bcc.cif`。本 adaptor 将 `/workspace/` 映射为当前 session 的 `workspace_path`，即 `workspace_path/Fe_bcc.cif`，再判断文件是否存在并上传 OSS。

## 依赖

- 环境变量：`OSS_ENDPOINT`、`OSS_BUCKET_NAME` 及 OSS 鉴权。
- 可选：`pip install oss2` 或 `pip install -e ".[calculation]"`。
