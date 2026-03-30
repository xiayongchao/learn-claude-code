#!/usr/bin/env python3
# Harness: context isolation -- protecting the model's clarity of thought.
"""
s04_subagent.py - Subagents

Spawn a child agent with fresh messages=[]. The child works in its own
context, sharing the filesystem, then returns only a summary to the parent.

    Parent agent                     Subagent
    +------------------+             +------------------+
    | messages=[...]   |             | messages=[]      |  <-- fresh
    |                  |  dispatch   |                  |
    | tool: task       | ---------->| while tool_use:  |
    |   prompt="..."   |            |   call tools     |
    |   description="" |            |   append results |
    |                  |  summary   |                  |
    |   result = "..." | <--------- | return last text |
    +------------------+             +------------------+
              |
    Parent context stays clean.
    Subagent context is discarded.

Key insight: "Process isolation gives context isolation for free."
"""

import os
import subprocess
from pathlib import Path

import ast
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path.cwd()
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
MODEL = os.environ.get("MODEL_ID", "qwen3.5-plus")

SYSTEM = f"你是工作目录 `{WORKDIR}` 下的代码智能体，请使用任务工具来分派代码调研工作或各类子任务。"
SUBAGENT_SYSTEM = f"你是运行在 **{WORKDIR}** 工作目录下的编码子智能体。完成指定任务后，请汇总你的调研结果与结论。"


# -- Tool implementations shared by parent and child --

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"路径超出工作目录范围: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "错误：已拦截危险命令"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(无输出)"
    except subprocess.TimeoutExpired:
        return "错误：超时 (120s)"


def run_read(path: str, limit: int = None) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"错误: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return  f"已向 {path} 写入 {len(content.encode('utf-8'))} 字节数据"
    except Exception as e:
        return f"错误: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"错误：在 {path} 中未找到对应文本"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"错误: {e}"


TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

# Child gets all base tools except task (no recursive spawning)
CHILD_TOOLS = [
       {"type": "function", "function": {"name": "bash", "description": "执行一条 Shell 命令",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "读取文件内容",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "写入文件内容",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    
    {"type": "function", "function": {"name": "edit_file", "description": "精确替换文件中的文本内容。",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},

  ]


# -- Subagent: fresh context, filtered tools, summary-only return --
def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]  # fresh context
    for _ in range(30):  # safety limit
        full_messages = [{"role": "system", "content": SUBAGENT_SYSTEM}] + sub_messages
        response = client.chat.completions.create(
            model=MODEL,
            messages=full_messages,
            tools=CHILD_TOOLS,
            max_tokens=4096,
        )
        message = response.choices[0].message
        
        assistant_msg = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in message.tool_calls
            ]
        sub_messages.append(assistant_msg)
        
        if response.choices[0].finish_reason != "tool_calls":
            break
        
        if message.tool_calls:
            for tool_call in message.tool_calls:
                handler = TOOL_HANDLERS.get(tool_call.function.name)
                import json
                try:
                    
                    args = ast.literal_eval(tool_call.function.arguments)
                    output = handler(**args) if handler else f"未知工具: {tool_call.function.name}"
                    
                except ValueError as e:
                    try:
                        args = json.loads(tool_call.function.arguments)
                        output = handler(**args) if handler else f"未知工具: {tool_call.function.name}"
                    except json.JSONDecodeError as e2:
                        output = f"JSON错误: {e2}"
                except Exception as e:
                    output = f"错误: {e}"
                sub_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(output)[:50000]
                })
    # Only the final text returns to the parent -- child context is discarded
    return message.content or "(no summary)"


# -- Parent tools: base tools + task dispatcher --
PARENT_TOOLS = CHILD_TOOLS + [
    {"type": "function", "function": {"name": "task", "description": "生成一个携带全新上下文的子智能体。它共享文件系统，但不共享对话历史。",
     "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}, "description": {"type": "string", "description": "任务简要说明"}}, "required": ["prompt"]}}},
]


def agent_loop(messages: list):
    while True:
        full_messages = [{"role": "system", "content": SYSTEM}] + messages
        response = client.chat.completions.create(
            model=MODEL,
            messages=full_messages,
            tools=PARENT_TOOLS,
            max_tokens=4096,
        )
        message = response.choices[0].message
        
        assistant_msg = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in message.tool_calls
            ]
        messages.append(assistant_msg)
        
        if response.choices[0].finish_reason != "tool_calls":
            return
        
        if message.tool_calls:
            for tool_call in message.tool_calls:
                handler = TOOL_HANDLERS.get(tool_call.function.name)
                import json
                try:
                    args = ast.literal_eval(tool_call.function.arguments)
                    if tool_call.function.name == "task":
                        desc = args.get("description", "subtask")
                        print(f"> 子任务任务 ({desc}): {args['prompt'][:80]}")
                        output = run_subagent(args["prompt"])
                    else:
                        output = handler(**args) if handler else f"未知工具: {tool_call.function.name}"
                except ValueError as e:
                    # 解析失败 → 尝试用标准 JSON 再解析一次
                    try:
                        args = json.loads(tool_call.function.arguments)
                        if tool_call.function.name == "task":
                            desc = args.get("description", "subtask")
                            print(f"> 子任务任务 ({desc}): {args['prompt'][:80]}")
                            output = run_subagent(args["prompt"])
                        else:
                            output = handler(**args) if handler else f"未知工具: {tool_call.function.name}"
                    except json.JSONDecodeError as e2:
                        output = f"JSON错误: {e2}"
                except Exception as e:
                    output = f"错误: {e}"
                print(f"  {str(output)[:200]}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(output)
                })


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms04 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if response_content:
            print(response_content)
        print()
