#!/usr/bin/env python3
# 工具：所有机制组合 -- 模型的完整驾驶舱
"""
s_full_zh.py - 完整参考代理

结合 s01-s11 所有机制的顶点实现。
s12（任务感知工作树隔离）单独教授。
不是教学会话 -- 这是"把所有东西放在一起"的参考。

    +------------------------------------------------------------------+
    |                        完整代理                                   |
    |                                                                   |
    |  系统提示（s05 技能，任务优先 + 可选待办事项提醒）                |
    |                                                                   |
    |  每次 LLM 调用前：                                                |
    |  +--------------------+  +------------------+  +--------------+  |
    |  | 微压缩（s06）      |  | 后台通知（s08）  |  | 检查收件箱   |  |
    |  | 自动压缩（s06）    |  | 排水             |  | （s09）      |  |
    |  +--------------------+  +------------------+  +--------------+  |
    |                                                                   |
    |  工具调度（s02 模式）：                                           |
    |  +--------+----------+----------+---------+-----------+          |
    |  | bash   | read     | write    | edit    | TodoWrite |          |
    |  | task   | load_sk  | compress | bg_run  | bg_check  |          |
    |  | t_crt  | t_get    | t_upd    | t_list  | spawn_tm  |          |
    |  | list_tm| send_msg | rd_inbox | bcast   | shutdown  |          |
    |  | plan   | idle     | claim    |         |           |          |
    |  +--------+----------+----------+---------+-----------+          |
    |                                                                   |
    |  子代理（s04）： 生成 -> 工作 -> 返回摘要                        |
    |  队友（s09）： 生成 -> 工作 -> 空闲 -> 自动认领（s11）           |
    |  关闭（s10）： request_id 握手                                   |
    |  计划门（s10）： 提交 -> 批准/拒绝                               |
    +------------------------------------------------------------------+

    REPL 命令：/compact /tasks /team /inbox
"""

import json
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from queue import Queue

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path.cwd()

# 千问模型配置
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
)

MODEL = os.environ.get("MODEL_ID", "qwen-max")

TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"
TASKS_DIR = WORKDIR / ".tasks"
SKILLS_DIR = WORKDIR / "skills"
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TOKEN_THRESHOLD = 100000
POLL_INTERVAL = 5
IDLE_TIMEOUT = 60

VALID_MESSAGE_TYPES = {"message", "broadcast", "shutdown_request",
                   "shutdown_response", "plan_approval_response"}


# === 部分：基础工具 ===
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"路径超出工作区：{p}")
    return path

def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "错误：危险命令被阻止"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(无输出)"
    except subprocess.TimeoutExpired:
        return "错误：超时 (120s)"

def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} 更多)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"错误：{e}"

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"已写入 {len(content)} 字节到 {path}"
    except Exception as e:
        return f"错误：{e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"错误：在 {path} 中未找到文本"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"已编辑 {path}"
    except Exception as e:
        return f"错误：{e}"


# === 部分：待办事项（s03）===
class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        validated, in_progress = [], 0
        for i, item in enumerate(items):
            content = str(item.get("content", "")).strip()
            status = str(item.get("status", "pending")).lower()
            active_form = str(item.get("activeForm", "")).strip()
            if not content: raise ValueError(f"项目 {i}：需要内容")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"项目 {i}：无效状态 '{status}'")
            if not active_form: raise ValueError(f"项目 {i}：需要 activeForm")
            if status == "in_progress": in_progress += 1
            validated.append({"content": content, "status": status, "activeForm": active_form})
        if len(validated) > 20: raise ValueError("最多 20 个待办事项")
        if in_progress > 1: raise ValueError("只允许一个进行中")
        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items: return "无待办事项。"
        lines = []
        for item in self.items:
            m = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]"}.get(item["status"], "[?]")
            suffix = f" <- {item['activeForm']}" if item["status"] == "in_progress" else ""
            lines.append(f"{m} {item['content']}{suffix}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} 已完成)")
        return "\n".join(lines)

    def has_unfinished(self) -> bool:
        return any(item.get("status") != "completed" for item in self.items)


# === 部分：子代理（s04）===
def run_subagent(prompt: str, agent_type: str = "Explore") -> str:
    sub_tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "运行命令。",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "读取文件。",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
            }
        },
    ]
    if agent_type != "Explore":
        sub_tools += [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "写入文件。",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "编辑文件。",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}
                }
            },
        ]
    sub_handlers = {
        "bash": lambda **kw: run_bash(kw["command"]),
        "read_file": lambda **kw: run_read(kw["path"]),
        "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
        "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    }
    sub_messages = [{"role": "user", "content": prompt}]
    response = None
    for _ in range(30):
        response = client.chat.completions.create(model=MODEL, messages=sub_messages, tools=sub_tools, max_tokens=8000)
        sub_messages.append({"role": "assistant", "content": ""})
        if response.choices[0].finish_reason != "tool_calls":
            sub_messages[-1]["content"] = response.choices[0].message.content
            break
        tool_calls = response.choices[0].message.tool_calls
        sub_messages[-1]["tool_calls"] = tool_calls
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            h = sub_handlers.get(tool_name, lambda **kw: "未知工具")
            results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(h(**tool_args))[:50000]
            })
        sub_messages.extend(results)
    if response:
        return response.choices[0].message.content or "(无摘要)"
    return "(子代理失败)"


# === 部分：技能（s05）===
class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills = {}
        if skills_dir.exists():
            for f in sorted(skills_dir.rglob("SKILL.md")):
                text = f.read_text()
                match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
                meta, body = {}, text
                if match:
                    for line in match.group(1).strip().splitlines():
                        if ":" in line:
                            k, v = line.split(":", 1)
                            meta[k.strip()] = v.strip()
                    body = match.group(2).strip()
                name = meta.get("name", f.parent.name)
                self.skills[name] = {"meta": meta, "body": body}

    def describe(self) -> str:
        if not self.skills: return "(无技能)"
        return "\n".join(f"  - {n}: {s['meta'].get('description', '-')}" for n, s in self.skills.items())

    def load(self, name: str) -> str:
        s = self.skills.get(name)
        if not s: return f"错误：未知技能 '{name}'。可用：{', '.join(self.skills.keys())}"
        return f"<skill name=\"{name}\">\n{s['body']}\n</skill>"


# === 部分：压缩（s06）===
def estimate_tokens(messages: list) -> int:
    return len(json.dumps(messages, default=str)) // 4

def micro_compact(messages: list):
    indices = []
    for i, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    indices.append(part)
    if len(indices) <= 3:
        return
    for part in indices[:-3]:
        if isinstance(part.get("content"), str) and len(part["content"]) > 100:
            part["content"] = "[已清除]"

def auto_compact(messages: list) -> list:
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    dialog_text = json.dumps(messages, default=str)[:80000]
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"为连续性总结：\n{dialog_text}"}],
        max_tokens=2000,
    )
    summary = response.choices[0].message.content
    return [
        {"role": "user", "content": f"[已压缩。转录：{path}]\n{summary}"},
        {"role": "assistant", "content": "理解。继续使用摘要上下文。"},
    ]


# === 部分：文件任务（s07）===
class TaskManager:
    def __init__(self):
        TASKS_DIR.mkdir(exist_ok=True)

    def _next_id(self) -> int:
        ids = [int(f.stem.split("_")[1]) for f in TASKS_DIR.glob("task_*.json")]
        return max(ids, default=0) + 1

    def _load(self, task_id: int) -> dict:
        p = TASKS_DIR / f"task_{task_id}.json"
        if not p.exists(): raise ValueError(f"任务 {task_id} 未找到")
        return json.loads(p.read_text())

    def _save(self, task: dict):
        (TASKS_DIR / f"task_{task['id']}.json").write_text(json.dumps(task, indent=2, ensure_ascii=False))

    def create(self, subject: str, description: str = "") -> str:
        task = {"id": self._next_id(), "subject": subject, "description": description,
                "status": "pending", "owner": None, "blockedBy": [], "blocks": []}
        self._save(task)
        return json.dumps(task, indent=2, ensure_ascii=False)

    def get(self, task_id: int) -> str:
        return json.dumps(self._load(task_id), indent=2, ensure_ascii=False)

    def update(self, task_id: int, status: str = None,
               add_blocked_by: list = None, add_blocks: list = None) -> str:
        task = self._load(task_id)
        if status:
            task["status"] = status
            if status == "completed":
                for f in TASKS_DIR.glob("task_*.json"):
                    t = json.loads(f.read_text())
                    if task_id in t.get("blockedBy", []):
                        t["blockedBy"].remove(task_id)
                        self._save(t)
            if status == "deleted":
                (TASKS_DIR / f"task_{task_id}.json").unlink(missing_ok=True)
                return f"任务 {task_id} 已删除"
        if add_blocked_by:
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))
        if add_blocks:
            task["blocks"] = list(set(task["blocks"] + add_blocks))
        self._save(task)
        return json.dumps(task, indent=2, ensure_ascii=False)

    def list_all(self) -> str:
        tasks = [json.loads(f.read_text()) for f in sorted(TASKS_DIR.glob("task_*.json"))]
        if not tasks: return "无任务。"
        lines = []
        for t in tasks:
            m = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
            owner = f" @{t['owner']}" if t.get("owner") else ""
            blocked = f" (被阻塞：{t['blockedBy']})" if t.get("blockedBy") else ""
            lines.append(f"{m} #{t['id']}: {t['subject']}{owner}{blocked}")
        return "\n".join(lines)

    def claim(self, task_id: int, owner: str) -> str:
        task = self._load(task_id)
        task["owner"] = owner
        task["status"] = "in_progress"
        self._save(task)
        return f"已为 {owner} 认领任务 #{task_id}"


# === 部分：后台（s08）===
class BackgroundManager:
    def __init__(self):
        self.tasks = {}
        self.notifications = Queue()

    def run(self, command: str, timeout: int = 120) -> str:
        task_id = str(uuid.uuid4())[:8]
        self.tasks[task_id] = {"status": "running", "command": command, "result": None}
        threading.Thread(target=self._execute, args=(task_id, command, timeout), daemon=True).start()
        return f"后台任务 {task_id} 已启动：{command[:80]}"

    def _execute(self, task_id: str, command: str, timeout: int):
        try:
            r = subprocess.run(command, shell=True, cwd=WORKDIR,
                               capture_output=True, text=True, timeout=timeout)
            out = (r.stdout + r.stderr).strip()[:50000]
            self.tasks[task_id].update({"status": "completed", "result": out or "(无输出)"})
        except Exception as e:
            self.tasks[task_id].update({"status": "error", "result": str(e)})
        self.notifications.put({"task_id": task_id, "status": self.tasks[task_id]["status"],
                        "result": self.tasks[task_id]["result"][:500]})

    def check(self, task_id: str = None) -> str:
        if task_id:
            t = self.tasks.get(task_id)
            return f"[{t['status']}] {t.get('result', '(运行中)'}" if t else f"未知：{task_id}"
        return "\n".join(f"{k}: [{v['status']}] {v['command'][:60]}" for k, v in self.tasks.items()) or "无后台任务。"

    def drain(self) -> list:
        notifications = []
        while not self.notifications.empty():
            notifications.append(self.notifications.get_nowait())
        return notifications


# === 部分：消息传递（s09）===
class MessageBus:
    def __init__(self):
        INBOX_DIR.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, receiver: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        msg = {"type": msg_type, "from": sender, "content": content,
               "timestamp": time.time()}
        if extra: msg.update(extra)
        with open(INBOX_DIR / f"{receiver}.jsonl", "a") as f:
            f.write(json.dumps(msg) + "\n")
        return f"已发送 {msg_type} 到 {receiver}"

    def read_inbox(self, name: str) -> list:
        path = INBOX_DIR / f"{name}.jsonl"
        if not path.exists(): return []
        messages = [json.loads(l) for l in path.read_text().strip().splitlines() if l]
        path.write_text("")
        return messages

    def broadcast(self, sender: str, content: str, names: list) -> str:
        count = 0
        for n in names:
            if n != sender:
                self.send(sender, n, content, "broadcast")
                count += 1
        return f"已广播到 {count} 个队友"


# === 部分：关闭 + 计划跟踪（s10）===
shutdown_requests = {}
plan_requests = {}


# === 部分：团队（s09/s11）===
class TeammateManager:
    def __init__(self, bus: MessageBus, task_manager: TaskManager):
        TEAM_DIR.mkdir(exist_ok=True)
        self.bus = bus
        self.task_manager = task_manager
        self.config_path = TEAM_DIR / "config.json"
        self.config = self._load()
        self.threads = {}

    def _load(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save(self):
        self.config_path.write_text(json.dumps(self.config, indent=2, ensure_ascii=False))

    def _find(self, name: str) -> dict:
        for m in self.config["members"]:
            if m["name"] == name: return m
        return None

    def spawn(self, name: str, role: str, prompt: str) -> str:
        member = self._find(name)
        if member:
            if member["status"] not in ("idle", "shutdown"):
                return f"错误：'{name}' 当前状态为 {member['status']}"
            member["status"] = "working"
            member["role"] = role
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save()
        threading.Thread(target=self._loop, args=(name, role, prompt), daemon=True).start()
        return f"已生成 '{name}' (角色: {role})"

    def _set_status(self, name: str, status: str):
        member = self._find(name)
        if member:
            member["status"] = status
            self._save()

    def _loop(self, name: str, role: str, prompt: str):
        team_name = self.config["team_name"]
        messages = [
            {"role": "system", "content": f"你是 '{name}', 角色: {role}, 团队: {team_name}, 在 {WORKDIR}。 完成当前工作后使用 idle。你可以自动认领任务。"},
            {"role": "user", "content": prompt}
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "运行命令。",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "读取文件。",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "写入文件。",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "编辑文件。",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "send_message",
                    "description": "发送消息。",
                    "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}}, "required": ["to", "content"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "idle",
                    "description": "表示无更多工作。",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "claim_task",
                    "description": "按 ID 认领任务。",
                    "parameters": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}
                }
            },
        ]
        while True:
            # -- 工作阶段 --
            for _ in range(50):
                inbox = self.bus.read_inbox(name)
                for msg in inbox:
                    if msg.get("type") == "shutdown_request":
                        self._set_status(name, "shutdown")
                        return
                    messages.append({"role": "user", "content": json.dumps(msg)})
                try:
                    response = client.chat.completions.create(
                        model=MODEL, messages=messages,
                        tools=tools, max_tokens=8000)
                except Exception:
                    self._set_status(name, "shutdown")
                    return
                messages.append({"role": "assistant", "content": ""})
                if response.choices[0].finish_reason != "tool_calls":
                    messages[-1]["content"] = response.choices[0].message.content
                    break
                tool_calls = response.choices[0].message.tool_calls
                messages[-1]["tool_calls"] = tool_calls
                results = []
                request_idle = False
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    if tool_name == "idle":
                        request_idle = True
                        output = "进入空闲阶段。"
                    elif tool_name == "claim_task":
                        output = self.task_manager.claim(tool_args["task_id"], name)
                    elif tool_name == "send_message":
                        output = self.bus.send(name, tool_args["to"], tool_args["content"])
                    else:
                        dispatch = {"bash": lambda **kw: run_bash(kw["command"]),
                                "read_file": lambda **kw: run_read(kw["path"]),
                                "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
                                "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"])}
                        output = dispatch.get(tool_name, lambda **kw: "未知")(**tool_args)
                    print(f"  [{name}] {tool_name}: {str(output)[:120]}")
                    results.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(output)
                    })
                messages.extend(results)
                if request_idle:
                    break
            # -- 空闲阶段：轮询消息和未认领任务 --
            self._set_status(name, "idle")
            resume = False
            for _ in range(IDLE_TIMEOUT // max(POLL_INTERVAL, 1)):
                time.sleep(POLL_INTERVAL)
                inbox = self.bus.read_inbox(name)
                if inbox:
                    for msg in inbox:
                        if msg.get("type") == "shutdown_request":
                            self._set_status(name, "shutdown")
                            return
                        messages.append({"role": "user", "content": json.dumps(msg)})
                    resume = True
                    break
                unclaimed = []
                for f in sorted(TASKS_DIR.glob("task_*.json")):
                    t = json.loads(f.read_text())
                    if t.get("status") == "pending" and not t.get("owner") and not t.get("blockedBy"):
                        unclaimed.append(t)
                if unclaimed:
                    task = unclaimed[0]
                    self.task_manager.claim(task["id"], name)
                    # 身份重新注入以压缩上下文
                    if len(messages) <= 3:
                        messages.insert(0, {"role": "user", "content":
                            f"<identity>你是 '{name}', 角色: {role}, 团队: {team_name}.</identity>"})
                        messages.insert(1, {"role": "assistant", "content": f"我是 {name}。继续。"})
                    messages.append({"role": "user", "content":
                        f"<auto-claimed>任务 #{task['id']}: {task['subject']}\n{task.get('description', '')}</auto-claimed>"})
                    messages.append({"role": "assistant", "content": f"已认领任务 #{task['id']}。正在处理。"})
                    resume = True
                    break
            if not resume:
                self._set_status(name, "shutdown")
                return
            self._set_status(name, "working")

    def list_all(self) -> str:
        if not self.config["members"]: return "无队友。"
        lines = [f"团队: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        return [m["name"] for m in self.config["members"]]


# === 部分：全局实例 ===
todo = TodoManager()
skills = SkillLoader(SKILLS_DIR)
task_manager = TaskManager()
background = BackgroundManager()
bus = MessageBus()
team = TeammateManager(bus, task_manager)

# === 部分：系统提示 ===
system_prompt = f"""你是 {WORKDIR} 的编码代理。使用工具解决任务。
优先使用 task_create/task_update/task_list 进行多步骤工作。使用 TodoWrite 进行简短清单。
使用 task 进行子代理委托。使用 load_skill 获取专业知识。
技能：{skills.describe()}"""


# === 部分：关闭协议（s10）===
def handle_shutdown_request(teammate: str) -> str:
    request_id = str(uuid.uuid4())[:8]
    shutdown_requests[request_id] = {"target": teammate, "status": "pending"}
    bus.send("lead", teammate, "请关闭。", "shutdown_request", {"request_id": request_id})
    return f"关闭请求 {request_id} 已发送到 '{teammate}'"

# === 部分：计划批准（s10）===
def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    req = plan_requests.get(request_id)
    if not req: return f"错误：未知计划请求ID '{request_id}'"
    req["status"] = "approved" if approve else "rejected"
    bus.send("lead", req["from"], feedback, "plan_approval_response",
             {"request_id": request_id, "approve": approve, "feedback": feedback})
    return f"计划 {req['status']} 来自 '{req['from']}'"


# === 部分：工具调度（s02）===
tool_handlers = {
    "bash":             lambda **kw: run_bash(kw["command"]),
    "read_file":        lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":       lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":        lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "TodoWrite":        lambda **kw: todo.update(kw["items"]),
    "task":             lambda **kw: run_subagent(kw["prompt"], kw.get("agent_type", "Explore")),
    "load_skill":       lambda **kw: skills.load(kw["name"]),
    "compress":         lambda **kw: "压缩中...",
    "background_run":   lambda **kw: background.run(kw["command"], kw.get("timeout", 120)),
    "check_background": lambda **kw: background.check(kw.get("task_id")),
    "task_create":      lambda **kw: task_manager.create(kw["subject"], kw.get("description", "")),
    "task_get":         lambda **kw: task_manager.get(kw["task_id"]),
    "task_update":      lambda **kw: task_manager.update(kw["task_id"], kw.get("status"), kw.get("add_blocked_by"), kw.get("add_blocks")),
    "task_list":        lambda **kw: task_manager.list_all(),
    "spawn_teammate":   lambda **kw: team.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates":   lambda **kw: team.list_all(),
    "send_message":     lambda **kw: bus.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":       lambda **kw: json.dumps(bus.read_inbox("lead"), indent=2),
    "broadcast":        lambda **kw: bus.broadcast("lead", kw["content"], team.member_names()),
    "shutdown_request": lambda **kw: handle_shutdown_request(kw["teammate"]),
    "plan_approval":    lambda **kw: handle_plan_review(kw["request_id"], kw["approve"], kw.get("feedback", "")),
    "idle":             lambda **kw: "领导不空闲。",
    "claim_task":       lambda **kw: task_manager.claim(kw["task_id"], "lead"),
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "运行 shell 命令。",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容。",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "向文件写入内容。",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "替换文件中的精确文本。",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "TodoWrite",
            "description": "更新任务跟踪列表。",
            "parameters": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"content": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}, "activeForm": {"type": "string"}}, "required": ["content", "status", "activeForm"]}}}, "required": ["items"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": "生成子代理进行隔离探索或工作。",
            "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}, "agent_type": {"type": "string", "enum": ["Explore", "general-purpose"]}}, "required": ["prompt"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": "按名称加载专业知识。",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compress",
            "description": "手动压缩对话上下文。",
            "parameters": {"type": "object", "properties": {}}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "background_run",
            "description": "在后台线程中运行命令。",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_background",
            "description": "检查后台任务状态。",
            "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task_create",
            "description": "创建持久文件任务。",
            "parameters": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task_get",
            "description": "按 ID 获取任务详情。",
            "parameters": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task_update",
            "description": "更新任务状态或依赖项。",
            "parameters": {"type": "object", "properties": {"task_id": {"type": "integer"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "deleted"]}, "add_blocked_by": {"type": "array", "items": {"type": "integer"}}, "add_blocks": {"type": "array", "items": {"type": "integer"}}}, "required": ["task_id"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task_list",
            "description": "列出所有任务。",
            "parameters": {"type": "object", "properties": {}}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_teammate",
            "description": "生成持久自主队友。",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["name", "role", "prompt"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_teammates",
            "description": "列出所有队友。",
            "parameters": {"type": "object", "properties": {}}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "向队友发送消息。",
            "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MESSAGE_TYPES)}}, "required": ["to", "content"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_inbox",
            "description": "读取并清空领导的收件箱。",
            "parameters": {"type": "object", "properties": {}}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "broadcast",
            "description": "向所有队友发送消息。",
            "parameters": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "shutdown_request",
            "description": "请求队友关闭。",
            "parameters": {"type": "object", "properties": {"teammate": {"type": "string"}}, "required": ["teammate"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plan_approval",
            "description": "批准或拒绝队友的计划。",
            "parameters": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "feedback": {"type": "string"}}, "required": ["request_id", "approve"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "idle",
            "description": "进入空闲状态。",
            "parameters": {"type": "object", "properties": {}}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "claim_task",
            "description": "从任务板认领任务。",
            "parameters": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}
        }
    },
]


# === 部分：代理循环 ===
def agent_loop(messages: list):
    no_todo_rounds = 0
    # 确保系统提示在消息列表的开头
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": system_prompt})
    while True:
        # s06: 压缩管道
        micro_compact(messages)
        if estimate_tokens(messages) > TOKEN_THRESHOLD:
            print("[自动压缩触发]")
            compacted = auto_compact(messages[1:])  # 排除系统提示
            messages = [messages[0]] + compacted  # 重新添加系统提示
        # s08: 排空后台通知
        notifications = background.drain()
        if notifications:
            text = "\n".join(f"[bg:{n['task_id']}] {n['status']}: {n['result']}" for n in notifications)
            messages.append({"role": "user", "content": f"<background-results>\n{text}\n</background-results>"})
            messages.append({"role": "assistant", "content": "已注意到后台结果。"})
        # s10: 检查领导收件箱
        inbox = bus.read_inbox("lead")
        if inbox:
            messages.append({"role": "user", "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>"})
            messages.append({"role": "assistant", "content": "已注意到收件箱消息。"})
        # LLM 调用
        response = client.chat.completions.create(
            model=MODEL, messages=messages,
            tools=tools, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": ""})
        if response.choices[0].finish_reason != "tool_calls":
            messages[-1]["content"] = response.choices[0].message.content
            return
        # 工具执行
        tool_calls = response.choices[0].message.tool_calls
        messages[-1]["tool_calls"] = tool_calls
        results = []
        used_todo = False
        manual_compact = False
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            if tool_name == "TodoWrite":
                used_todo = True
            if tool_name == "compress":
                manual_compact = True
            
            handler = tool_handlers.get(tool_name)
            try:
                output = handler(**tool_args) if handler else f"未知工具：{tool_name}"
            except Exception as e:
                output = f"错误：{e}"
            print(f"> {tool_name}：{str(output)[:200]}")
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(output),
                }
            )
        messages.extend(results)
        
        if manual_compact:
            print("[手动压缩触发]")
            compacted = auto_compact(messages[1:])  # 排除系统提示
            messages = [messages[0]] + compacted  # 重新添加系统提示
        
        if not used_todo and todo.has_unfinished():
            no_todo_rounds += 1
            if no_todo_rounds >= 3:
                messages.append({"role": "user", "content": f"<todo-reminder>\n{todo.render()}\n</todo-reminder>"})
                messages.append({"role": "assistant", "content": "已注意到待办事项。"})
                no_todo_rounds = 0
        else:
            no_todo_rounds = 0


if __name__ == "__main__":
    print(f"完整代理在 {WORKDIR}")
    
    history = []
    while True:
        try:
            query = input("\033[36ms_full >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip().startswith("/"):
            cmd = query.strip()[1:].lower()
            if cmd == "compact":
                print("[手动压缩触发]")
                if len(history) > 1:
                    compacted = auto_compact(history[1:])  # 排除系统提示
                    history = [history[0]] + compacted  # 重新添加系统提示
                print("完成。")
                continue
            elif cmd == "tasks":
                print(todo.render())
                continue
            elif cmd == "team":
                print(team.list_all())
                continue
            elif cmd == "inbox":
                inbox = bus.read_inbox("lead")
                print(json.dumps(inbox, indent=2) if inbox else "收件箱为空。")
                continue
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1].get("content", "")
        if response_content:
            print(response_content)
        print()