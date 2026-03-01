# EvoMaster v0.0.2 版本更新说明

> 对比基线：`main` 分支 → `xinyu/parallel` 分支
> 涉及文件：143 个文件，+8968 / -1675 行

---

## 一、架构级变更

### 1. 配置系统重构：从全局单 LLM 到 per-agent 配置

旧架构中 `agent`（单数）只能关联一套全局 LLM；新版改为 `agents`（复数），每个 agent 独立声明自己的 LLM、Tools、Skills。

| 维度 | main (旧) | xinyu/parallel (新) |
|------|-----------|---------------------|
| 配置字段 | `agent: {...}` | `agents: { name: {...} }` |
| LLM 绑定 | 全局 `llm.default` | 每个 agent 可指定 `llm: "openai"` |
| 工具启用 | `enable_tools: true/false` | `tools: { builtin: [...], mcp: "..." }` |
| Skills | 全局 `skills.enabled` | 每个 agent 可指定 `skills: ["rag"]` 或 `"*"` |

核心文件变更：
- `evomaster/config.py`：新增 `ToolConfig`、移除 `SkillConfig/KnowledgeSkillConfig/OperatorSkillConfig`；新增 `get_agent_config(name)`、`get_agents_config()`、`get_agent_llm_config(name)`、`get_agent_tools_config(name)`、`get_agent_skills_config(name)` 等 per-agent 配置读取方法。
- `EvoMasterConfig.agent` → `EvoMasterConfig.agents`（字段改名）
- `EvoMasterConfig.skill` → `EvoMasterConfig.tools`（ToolConfig 替代 SkillConfig）

### 2. Playground 基类重构：多 Agent 槽位系统

- 新增 `AgentSlots` 容器类，支持 `dict` 访问与属性访问（`self.agents.planning_agent`）。
- `self.agent`（单个）→ `self.agents`（AgentSlots），向后兼容：`self.agent = self.agents.get_random_agent()`。
- 基类 `setup()` 流程简化为：`_setup_session()` → `_setup_agents()` → `（_setup_exps()）`。
- 新增 `_setup_agents()` 方法：自动遍历 `agents` 配置，为每个 agent 创建实例并注册到 `self.agents`。
- 新增 `copy_agent()` 方法：深拷贝 Agent（独立 LLM + 独立上下文，共享 session/tools），支持并行实验。

### 3. 工具注册系统重构：支持按名称筛选

- 新增 `create_registry(builtin_names, skill_registry)` 函数，替代原 `create_default_registry()`。
- 支持 `builtin_names=["execute_bash", "finish"]` 精确控制注册哪些 builtin 工具。
- Agent 新增 `enabled_tool_names` 参数：控制暴露给 LLM 的工具列表，与"代码中可调用的工具"解耦。
- 所有工具始终注册到 registry（代码可手动调用），仅通过 `enabled_tool_names` 过滤 LLM 可见的工具。

### 4. Skills 统一化

- **移除** `KnowledgeSkill` 和 `OperatorSkill` 二元分类。
- **统一为** `Skill` 类（原 `OperatorSkill` 重命名）。
- `SkillMetaInfo` 移除 `skill_type` 字段。
- `SkillRegistry` 新增 `skills` 参数支持按名称过滤加载；新增 `create_subset()` 方法。

---

## 二、新增功能

### 1. 并行实验执行

- `evomaster/env/local.py`：新增 `ResourceAllocator` 类，支持按并行索引自动分配 GPU/CPU 资源。
- `LocalSession`：新增线程本地存储（`_thread_local`），支持 `set_parallel_index()`、`set_workspace_path()` 等线程安全操作。
- `LocalEnv`：新增 `setup_exp_workspace()` 方法，支持 `split_workspace_for_exp` 模式（每个实验独立工作目录）。
- `LocalSessionConfig`：新增 `parallel` 配置字段（`enabled`、`max_parallel`、`split_workspace_for_exp`）。
- `BasePlayground`：新增 `execute_parallel_tasks()` 方法（通过 `ThreadPoolExecutor` 并行执行多个 exp）。

### 2. 多模态支持（图片输入）

- `evomaster/utils/llm.py`：新增 `encode_image_to_base64()`、`get_image_media_type()`、`build_multimodal_content()` 函数。
- `evomaster/utils/types.py`：`BaseMessage.content` 类型扩展为 `str | list[dict] | None`，支持多模态内容块。
- `TaskInstance`：新增 `images: list[str]` 字段。
- `AnthropicLLM`：新增 `_convert_content_for_anthropic()` 静态方法，自动将 OpenAI 格式多模态内容转换为 Anthropic 格式。
- `ContextManager` / `SimpleTokenCounter`：适配多模态内容的 token 估算（图片 ~750 tokens）。
- `run.py`：新增 `--images` 命令行参数。

### 3. 飞书机器人接口

- 新增 `evomaster/interface/feishu/` 模块，包含：
  - `app.py`：FeishuBot 主类（生命周期管理：初始化 → 事件接收 → 解析 → 去重 → 调度）
  - `dispatcher.py`：任务调度器（支持 `/agent <name> <task>` 命令）
  - `sender.py`：消息发送（支持文本/卡片消息，流式更新）
  - `dedup.py`：消息去重
  - `event_handler.py`：事件解析
  - `config.py`：飞书配置
  - `client.py`：飞书 SDK 客户端创建

### 4. ML-Master Playground

- 新增 `playground/ml_master/`：面向机器学习竞赛（如 Kaggle）的完整工作流。
  - 三阶段 Exp：`DraftExp`（初稿）→ `DebugExp`（调试）→ `ImproveExp`（改进）。
  - UCT 搜索管理器（`core/utils/uct.py`）：基于 Monte Carlo Tree Search 的实验方案搜索。
  - 数据预览工具（`core/utils/data_preview.py`）：自动生成数据集概览。
  - 可视化模块（`vis/`）：树形结构 Web 可视化。

### 5. 多 Agent 并行 Playground 示例

- 新增 `playground/minimal_multi_agent_parallel/`：演示如何使用 `copy_agent()` + `ThreadPoolExecutor` 并行运行多个实验。

### 6. `extract_agent_response()` 模块级工具函数

- `evomaster/core/exp.py`：新增 `extract_agent_response()` 函数，支持从对象或 dict 格式的轨迹中提取 Agent 最终回答（优先提取 `finish` tool_call 的 message 参数）。

---

## 三、重要修改

### 配置文件格式变更

- `enable_tools: true/false` → `tools: { builtin: ["*"], mcp: "" }`
- 全局 `mcp:` 配置 → per-agent `tools.mcp` 配置（保留全局 `mcp:` 作为高级选项）
- `agent:` → `agents:` (配置文件中的字段名)

### Exp 结果格式变更（x_master）

- 结果从 `Dict[str, Any]`（如 `{"solver_result_0": "..."}`) 变为 `List[Dict]`（如 `[{"exp_index": 0, "solver_result": "..."}]`）。
- `_extract_solutions_from_results()` 等方法适配新格式。

### X-Master Playground

- `_setup_agents()` 使用新的 `_create_agent()` 签名（`llm_config` 替代 `llm_config_dict`）。
- Exp 不再在 `setup()` 中预创建，改为运行时通过 `_create_exp()` + `copy_agent()` 动态创建。
- 支持并行执行（移除了之前的 TODO 注释）。

### 依赖变更

- `requirements.txt` 新增飞书 SDK 等依赖。
- 新增 `.env.template` 模板文件。

---

## 四、移除 / 弃用

| 移除项 | 说明 |
|--------|------|
| `KnowledgeSkill` 类 | 统一为 `Skill` |
| `OperatorSkill` 类 | 统一为 `Skill` |
| `SkillConfig` / `KnowledgeSkillConfig` / `OperatorSkillConfig` | 被 per-agent skills 配置取代 |
| `ConfigManager.get_skill_config()` | 不再需要 |
| `BasePlayground._setup_llm_config()` | 被 `_setup_agent_llm(name)` 取代 |
| `_create_agent()` 的 `enable_tools` 和 `llm_config_dict` 参数 | 被 `tool_config` / `llm_config` / `skill_config` 取代 |
| `enable_tools: true/false` 配置字段 | 被 `tools: { builtin: [...] }` 取代 |

---

# 从 main 到 xinyu/parallel 的快速迁移指南

本指南帮助你将基于 `main` 分支编写的 Playground 迁移到 `xinyu/parallel` 分支的新架构。

## 第 1 步：迁移配置文件 (config.yaml)

### 1.1 `agent` → `agents`，每个 agent 声明 LLM

**旧写法 (main):**
```yaml
llm:
  openai:
    provider: "openai"
    model: "gpt-4"
    api_key: "${OPENAI_API_KEY}"
  default: "openai"

agents:
  planning:
    llm: "openai"
    max_turns: 10
    enable_tools: false     # <-- 旧写法
    context: ...
  coding:
    llm: "openai"
    max_turns: 50
    enable_tools: true      # <-- 旧写法
    context: ...
```

**新写法 (xinyu/parallel):**
```yaml
llm:
  openai:
    provider: "openai"
    model: "gpt-4"
    api_key: "${OPENAI_API_KEY}"
  default: "openai"

agents:
  planning:
    llm: "openai"           # per-agent LLM 绑定
    max_turns: 10
    tools:                   # <-- 新写法：精确控制工具
      builtin: []            # 不需要任何工具
    context: ...
  coding:
    llm: "openai"
    max_turns: 50
    tools:
      builtin: ["*"]         # 启用全部 builtin 工具
    context: ...
```

### 1.2 工具配置：`enable_tools` → `tools`

| 旧 | 新 | 说明 |
|----|-----|------|
| `enable_tools: true` | `tools: { builtin: ["*"] }` | 启用全部工具 |
| `enable_tools: false` | `tools: { builtin: [] }` | 禁用全部工具 |
| 无对应 | `tools: { builtin: ["execute_bash", "finish"] }` | 仅启用指定工具 |
| 全局 `mcp:` 配置 | `tools: { mcp: "mcp_config.json" }` | per-agent MCP |
| 不配置 `tools` 键 | 默认 `builtin: ["*"], mcp: ""` | 全部 builtin |

### 1.3 Skills 配置

**旧写法:** 全局配置
```yaml
skills:
  enabled: true
  skills_root: "evomaster/skills"
```

**新写法:** per-agent
```yaml
agents:
  coding:
    llm: "openai"
    skills: ["rag"]        # 仅加载 rag skill
    # 或 skills: "*"       # 加载全部 skills
```

## 第 2 步：迁移 Playground 代码

### 2.1 Agent 存储：从独立属性到 AgentSlots

**旧写法:**
```python
class MyPlayground(BasePlayground):
    def __init__(self, ...):
        super().__init__(...)
        self.planning_agent = None
        self.coding_agent = None
        self.mcp_manager = None

    def setup(self):
        llm_config_dict = self._setup_llm_config()
        self._setup_session()

        # 手动加载 skills
        skill_registry = None
        config_dict = self.config.model_dump()
        skills_config = config_dict.get("skills", {})
        if skills_config.get("enabled", False):
            skills_root = Path(skills_config.get("skills_root", "evomaster/skills"))
            skill_registry = SkillRegistry(skills_root)

        # 手动创建工具
        self._setup_tools(skill_registry)

        # 手动遍历 agents 配置并创建
        agents_config = getattr(self.config, 'agents', {})
        if 'planning' in agents_config:
            planning_config = agents_config['planning']
            self.planning_agent = self._create_agent(
                name="planning",
                agent_config=planning_config,
                enable_tools=planning_config.get('enable_tools', False),
                llm_config_dict=llm_config_dict,
                skill_registry=skill_registry,
            )
        if 'coding' in agents_config:
            coding_config = agents_config['coding']
            self.coding_agent = self._create_agent(
                name="coding",
                agent_config=coding_config,
                enable_tools=coding_config.get('enable_tools', True),
                llm_config_dict=llm_config_dict,
                skill_registry=skill_registry,
            )
```

**新写法:**
```python
class MyPlayground(BasePlayground):
    def __init__(self, ...):
        super().__init__(...)
        # 1. 声明 agent 槽位（IDE 补全友好）
        self.agents.declare("planning_agent", "coding_agent")
        self.mcp_manager = None

    def setup(self):
        # 2. 两行搞定！基类自动处理 LLM/Tools/Skills
        self._setup_session()
        self._setup_agents()
        # 基类自动将配置中的每个 agent 创建并注册到 self.agents
        # 命名规则: config 中的 "planning" → self.agents.planning_agent
```

### 2.2 `_create_agent()` 参数变更

**旧签名:**
```python
self._create_agent(
    name="solver",
    agent_config=solver_config,
    enable_tools=solver_config.get('enable_tools', False),
    llm_config_dict=llm_config_dict,
    skill_registry=skill_registry,
)
```

**新签名:**
```python
self._create_agent(
    name="solver",
    agent_config=solver_config,     # 可选，不传则自动从配置获取
    llm_config=llm_config,          # 可选，不传则自动从配置获取
    tool_config=tool_config,        # 可选，不传则自动从配置获取
    skill_config=skill_config,      # 可选，不传则自动从配置获取
)
```

如果使用 `_setup_agents()` 则无需手动调用 `_create_agent()`。

### 2.3 访问 Agent

**旧写法:**
```python
# 直接通过属性
self.planning_agent.run(task)
self.coding_agent.run(task)
```

**新写法:**
```python
# 通过 AgentSlots（同样支持属性访问）
self.agents.planning_agent.run(task)
self.agents.coding_agent.run(task)
```

### 2.4 Exp 中使用 Agent（并行场景）

**旧写法:** 直接使用共享的 agent 引用
```python
def _create_exp(self):
    exp = MultiAgentExp(
        planning_agent=self.planning_agent,
        coding_agent=self.coding_agent,
    )
    return exp
```

**新写法:** 使用 `copy_agent()` 创建独立副本（并行安全）
```python
def _create_exp(self, exp_index):
    planning_copy = self.copy_agent(
        self.agents.planning_agent,
        new_agent_name=f"planning_exp_{exp_index}"
    )
    coding_copy = self.copy_agent(
        self.agents.coding_agent,
        new_agent_name=f"coding_exp_{exp_index}"
    )
    exp = MultiAgentExp(
        planning_agent=planning_copy,
        coding_agent=coding_copy,
        exp_index=exp_index
    )
    return exp
```

## 第 3 步：迁移 Skills 引用

**旧写法:**
```python
from evomaster.skills import KnowledgeSkill, OperatorSkill

# 检查类型
if isinstance(skill, OperatorSkill):
    ...
elif isinstance(skill, KnowledgeSkill):
    ...
```

**新写法:**
```python
from evomaster.skills import Skill

# 统一类型，不再区分
if isinstance(skill, Skill):
    ...
```

## 第 4 步：迁移 Exp 结果提取（如适用）

**旧写法:** 结果是 dict，key 含索引
```python
results = {"solver_result_0": "...", "solver_result_1": "..."}
for i in range(self.agent_num):
    key = f"solver_result_{i}"
    if key in results:
        solutions.append(results[key])
```

**新写法:** 结果是 list，每个元素含 `exp_index`
```python
results = [{"exp_index": 0, "solver_result": "..."}, {"exp_index": 1, "solver_result": "..."}]
for result in results:
    key = "solver_result"
    if key in result and result[key] is not None:
        solutions.append(result[key])
```

## 迁移检查清单

- [ ] 配置文件：`enable_tools` → `tools: { builtin: [...] }`
- [ ] 配置文件：确认每个 agent 下有 `llm` 字段
- [ ] Playground：`self.xxx_agent = None` → `self.agents.declare("xxx_agent")`
- [ ] Playground：手动 `setup()` → `_setup_session()` + `_setup_agents()`
- [ ] Playground：`self.xxx_agent` → `self.agents.xxx_agent`
- [ ] 并行场景：使用 `copy_agent()` 创建独立副本
- [ ] Skills：`KnowledgeSkill/OperatorSkill` → `Skill`
- [ ] 移除对 `SkillConfig`、`get_skill_config()` 的引用
- [ ] 移除对 `_setup_llm_config()` 的调用（已被 per-agent 方法取代）
