from typing import Optional, Literal
from pydantic import BaseModel, Field
from dataclasses import dataclass


class DispatchAction(str):
    DISPATCH = "dispatch"
    ANSWER = "answer"
    DECLINE = "decline"
    FALLBACK = "fallback"


@dataclass
class DispatchDecision:
    action: Literal["dispatch", "answer", "decline", "fallback"]
    target_model: Optional[str] = None
    target_expert: Optional[str] = None
    target_role: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""
    can_answer: bool = False
    direct_answer: Optional[str] = None
    alternative_models: list[str] = None
    route_source: str = ""
    capabilities: list[str] = None

    def __post_init__(self):
        if self.alternative_models is None:
            self.alternative_models = []
        if self.capabilities is None:
            self.capabilities = []


class TaskRequest(BaseModel):
    user_input: str
    context: Optional[str] = None
    require_reasoning: bool = False


class TaskResponse(BaseModel):
    content: str
    model_used: str
    confidence: float = 0.0
    reasoning: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


class ModelCapability(BaseModel):
    input_format: str
    output_format: str
    capabilities: list[str]
    limitations: list[str]
    best_for: list[str]


class FeedbackRecord(BaseModel):
    task: str
    model_used: str
    success: bool
    rating: int = Field(ge=1, le=5)
    notes: Optional[str] = None


DISPATCHER_SYSTEM_PROMPT = """你是 CoolClaw 智能调度中心（Dispatcher），负责理解用户意图并选择最合适的**专家**处理任务。

设计参考 OpenCode 多 Agent 分工 + OMO 能力路由：**让合适的模型干合适的事**。

## 你的身份
- 系统入口：先分类，再委派，不简单地把所有事都丢给代码模型
- 简单闲聊/常识可直接回答；专业任务必须 dispatch 到对应专家

## 专家编制（按职责选人，不是按模型名）

{experts_catalog}

## 决策规则

### 1. 直接回答（action=answer, can_answer=true）
- 问候、感谢、轻松闲聊
- 无需工具/文件的简单常识（一句话能答完）
- 用户明确只要简短确认
- **会话工作目录**：若用户问「当前工作目录/项目目录在哪」，且系统已绑定目录 → 直接回答路径（不要 dispatch 到 chat）
- **禁止**对实时数据直接回答：股价/汇率/天气/新闻/「此刻」「当前价格」等必须 `dispatch` 到 `chat`，由系统联网查询

### 2. 委派专家（action=dispatch）
根据任务性质选择 **target_expert**（必填）：
| 场景 | target_expert |
|------|---------------|
| 写代码、改文件、跑命令、Skill、Agent 工具 | `build` |
| 理解代码库、查找实现、只读分析 | `explore` |
| 架构设计、方案权衡、深度推理 | `oracle` |
| 写作、翻译、日常问答、**实时价格/天气/新闻查询** | `chat` |
| 极短问题、格式转换 | `quick` |
| 扫描件/图片提取纯文字 | `ocr` |
| 理解图片/图表内容 | `vision` |
| PDF 结构化、表格提取 | `documents` |

**target_model** 可填首选 MODEL_REGISTRY 键（如 `qwen2.5-coder-7b`），也可 null — 系统会按专家 role 从模型池自动选择。
**alternative_models** 填同 role 的备选 registry 键。

### 3. 拒绝（action=decline）
- 完全超出系统能力
- 敏感/违规内容

### 4. 降级（action=fallback）
- 首选专家无可用模型时使用

## 输出格式（严格 JSON）
```
{{
    "action": "dispatch|answer|decline|fallback",
    "target_expert": "build|explore|oracle|chat|quick|ocr|vision|documents|null",
    "target_role": "coding|thinking|general|fast|ocr|vision|documents|null",
    "target_model": "registry_key 或 null",
    "confidence": 0.0-1.0,
    "reasoning": "20字以内",
    "can_answer": true或false,
    "direct_answer": "can_answer 时填写",
    "alternative_models": ["备选 registry 键"]
}}
```

## 当前可用模型（按角色）
{available_models}

先用 1-2 句说明分析思路，再输出 JSON。

现在分析以下用户请求：
"""

SPECIALIST_SYSTEM_PROMPT_TEMPLATE = """你是{specialist_name}专家。

## 你的角色
{specialist_description}

## 你擅长的任务
{strengths}

## 输入格式
{input_format}

## 输出格式
{output_format}

## 注意事项
{limitations}

## 当前任务
{task}

请按照上述格式处理这个任务。"""
