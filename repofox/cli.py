"""命令行入口。

这个模块负责把“用户怎么启动 repofox”翻译成 runtime 能理解的对象：
解析参数、挑模型后端、构建工作区快照、恢复或新建 session，
最后进入 one-shot 或交互式循环。
"""

import argparse
import json
import os
import shutil
import sys
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .models import AnthropicCompatibleModelClient, OllamaModelClient, OpenAICompatibleModelClient
from .runtime import RepoFox, SessionStore
from .workspace import WorkspaceContext, middle

DEFAULT_SECRET_ENV_NAMES = (
    "OPENAI_API_KEY",
    "OPENAI_API_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "RIGHT_CODES_API_KEY",
    "GITHUB_PAT",
    "GH_PAT",
)

WELCOME_ART = (
    "        /\\_/\\",
    "   ____/ o o \\",
    "  /  _ \\  ^  /",
    " /__/ \\_\\___/",
)
WELCOME_NAME = "RepoFox"
WELCOME_SUBTITLE = "local repo agent"
WELCOME_STATUS = "reading tracks, ready to patch"
HELP_DETAILS = textwrap.dedent(
    """\
    Commands:
    /help    Show this help message.
    /memory  Show the agent's distilled working memory.
    /session Show the path to the saved session file.
    /reset   Clear the current session history and memory.
    /exit    Exit the agent.
    """
).strip()


DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OPENAI_MODEL = "gpt-5.4"
DEFAULT_OPENAI_BASE_URL = "https://www.right.codes/codex/v1"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_ANTHROPIC_BASE_URL = "https://www.right.codes/claude/v1"
LEGACY_SECRET_ENV_NAMES_VAR = "MINI_CODING_AGENT_SECRET_ENV_NAMES"
SECRET_ENV_NAMES_VAR = "REPOFOX_SECRET_ENV_NAMES"


THINKING_FRAMES = ("|", "/", "-", "\\")
# CLI/SSE 状态事件只展示这些安全参数。`command`、`content` 等字段可能
# 包含敏感信息，所以不会出现在状态提示和 SSE 事件里。
SAFE_TOOL_ARG_KEYS = ("path", "pattern", "query", "start", "end")


def _thinking_label(elapsed_seconds, frame="|"):
    return f"Thinking {frame} {elapsed_seconds:.1f}s"


def _clip_status_text(text, limit=60):
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _tool_arg_summary(args):
    if not isinstance(args, dict):
        return ""
    parts = []
    for key in SAFE_TOOL_ARG_KEYS:
        value = args.get(key)
        if value in (None, ""):
            continue
        parts.append(f"{key}={_clip_status_text(value)}")
    return " ".join(parts)


def _safe_tool_args(args):
    if not isinstance(args, dict):
        return {}
    return {
        key: args[key]
        for key in SAFE_TOOL_ARG_KEYS
        if key in args and args[key] not in (None, "")
    }


def _sse_event(event, data):
    # 标准 SSE frame：一行 event、一行或多行 data，最后用空行结束。
    # 浏览器 EventSource 和 `curl -N` 都可以直接消费这种 text/event-stream。
    event = str(event).strip() or "message"
    if not isinstance(data, str):
        data = json.dumps(data, ensure_ascii=False, sort_keys=True)
    lines = [f"event: {event}"]
    for line in str(data).splitlines() or [""]:
        lines.append(f"data: {line}")
    lines.append("")
    return "\n".join(lines) + "\n"


class SSEAgentStream:
    """把 runtime 回调桥接成标准 SSE 事件 payload。"""

    def __init__(self, write_event, tick_interval=0.25, time_fn=None):
        self.write_event = write_event
        self.tick_interval = tick_interval
        self.time_fn = time_fn or time.monotonic
        self.thinking_started_at = None
        self.thinking_frame_index = 0
        self._thinking_thread = None
        self._thinking_stop = None
        self._lock = threading.Lock()
        self.emitted_final_delta = False

    def _emit(self, event, data):
        with self._lock:
            self.write_event(event, data)

    def _thinking_payload(self):
        started_at = self.thinking_started_at
        elapsed_ms = 0 if started_at is None else int((self.time_fn() - started_at) * 1000)
        return {
            "elapsed_ms": max(0, elapsed_ms),
            "frame": THINKING_FRAMES[self.thinking_frame_index],
            "status": "running",
        }

    def _thinking_loop(self, stop_event):
        # Runtime 每次请求模型前只发一次 thinking 事件。SSE 层额外维护一个
        # 轻量计时器，让模型阻塞等待期间，客户端仍能看到持续更新的进度。
        while not stop_event.wait(self.tick_interval):
            if self.thinking_started_at is None:
                return
            self.thinking_frame_index = (self.thinking_frame_index + 1) % len(THINKING_FRAMES)
            self._emit("thinking", self._thinking_payload())

    def start_thinking(self):
        self.stop_thinking()
        self.thinking_started_at = self.time_fn()
        self.thinking_frame_index = 0
        self._emit("thinking", self._thinking_payload())
        stop_event = threading.Event()
        self._thinking_stop = stop_event
        self._thinking_thread = threading.Thread(
            target=self._thinking_loop,
            args=(stop_event,),
            daemon=True,
        )
        self._thinking_thread.start()

    def stop_thinking(self):
        stop_event = self._thinking_stop
        thread = self._thinking_thread
        self._thinking_stop = None
        self._thinking_thread = None
        self.thinking_started_at = None
        if stop_event is not None:
            stop_event.set()
        if thread is not None:
            thread.join(timeout=0.2)

    def status_update(self, payload):
        # Runtime 状态事件是与传输方式无关的 dict。这里把它们映射成稳定的
        # SSE 事件名，并对工具参数做安全裁剪。
        event = str(payload.get("event", "")).strip()
        if event == "thinking":
            self.start_thinking()
            return
        self.stop_thinking()
        if event == "tool_started":
            args = _safe_tool_args(payload.get("args", {}))
            self._emit(
                "tool_started",
                {
                    "name": str(payload.get("name", "")).strip() or "unknown",
                    "args": args,
                    "summary": _tool_arg_summary(args),
                    "tool_steps": int(payload.get("tool_steps", 0) or 0),
                },
            )
            return
        if event == "tool_finished":
            self._emit(
                "tool_finished",
                {
                    "name": str(payload.get("name", "")).strip() or "unknown",
                    "duration_ms": int(payload.get("duration_ms", 0) or 0),
                },
            )

    def final_delta(self, text):
        if not text:
            return
        self.stop_thinking()
        self.emitted_final_delta = True
        self._emit("final_delta", {"text": text})

    def close(self):
        self.stop_thinking()


def _run_agent_sse(agent, prompt, write_event, stream_output=True):
    # 这是 HTTP handler 和测试共用的 “agent -> event stream” 适配器。
    # SSE 协议细节留在这一层，避免污染 RepoFox.ask() 的核心 runtime 逻辑。
    stream = SSEAgentStream(write_event)
    write_event(
        "run_started",
        {
            "session_id": agent.session["id"],
            "workspace": agent.workspace.cwd,
        },
    )
    try:
        final = agent.ask(
            prompt,
            stream_final=stream.final_delta if stream_output else None,
            status_update=stream.status_update,
        )
        if not stream.emitted_final_delta:
            stream.final_delta(final)
        write_event("final_answer", {"text": final})
        write_event("done", {"status": "ok"})
        return final
    except Exception as exc:
        write_event("error", {"message": str(exc), "type": exc.__class__.__name__})
        raise
    finally:
        stream.close()


class RepoFoxSSEHandler(BaseHTTPRequestHandler):
    """提供标准 Server-Sent Events 的 HTTP agent 运行入口。"""

    server_version = "RepoFoxSSE/0.1"

    def log_message(self, format, *args):
        return

    def _send_json_error(self, status, message):
        body = json.dumps({"error": message}, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _extract_prompt(self):
        parsed = urlparse(self.path)
        if parsed.path != "/ask":
            return None, "not_found"
        if self.command == "GET":
            values = parse_qs(parsed.query).get("prompt", [])
            prompt = values[0].strip() if values else ""
            return prompt, "" if prompt else "missing_prompt"
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return "", "invalid_json"
        prompt = str(payload.get("prompt", "")).strip()
        return prompt, "" if prompt else "missing_prompt"

    def _handle_ask(self):
        prompt, error = self._extract_prompt()
        if error == "not_found":
            self._send_json_error(404, "Not found. Use /ask.")
            return
        if error == "invalid_json":
            self._send_json_error(400, "Request body must be JSON.")
            return
        if error == "missing_prompt":
            self._send_json_error(400, "Request prompt is required.")
            return

        self.send_response(200)
        # 标准 SSE 响应头。X-Accel-Buffering 用于提示 nginx 类代理不要缓冲，
        # 否则前端可能无法实时收到事件。
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def write_event(event, data):
            self.wfile.write(_sse_event(event, data).encode("utf-8"))
            self.wfile.flush()

        try:
            _run_agent_sse(
                self.server.agent,
                prompt,
                write_event,
                stream_output=getattr(self.server, "stream_output", True),
            )
        except Exception:
            return

    def do_GET(self):
        self._handle_ask()

    def do_POST(self):
        self._handle_ask()


def serve_agent(agent, host="127.0.0.1", port=8765, stream_output=True):
    # 服务端刻意保持轻量：一个内存中的 agent 实例、一个 /ask 入口，
    # 以及 Python 标准库 HTTP server。
    server = ThreadingHTTPServer((host, int(port)), RepoFoxSSEHandler)
    server.agent = agent
    server.stream_output = bool(stream_output)
    print(f"RepoFox SSE server listening on http://{host}:{int(port)}/ask", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()


class ConsoleUI:
    """终端渲染器：负责 Copilot 风格状态行和最终答案增量输出。"""

    def __init__(self, fileobj, tick_interval=0.25, time_fn=None):
        self.fileobj = fileobj
        self.tick_interval = tick_interval
        self.time_fn = time_fn or time.monotonic
        self.lock = threading.Lock()
        self.wrote = False
        self.status_width = 0
        self.thinking_started_at = None
        self.thinking_frame_index = 0
        self._thinking_thread = None
        self._thinking_stop = None

    def _render_status(self, text):
        # 用回车符原地刷新同一行状态。日志里可能看到多个 \r，
        # 但交互式终端里会显示为一条实时更新的状态行。
        padded = text.ljust(self.status_width)
        self.status_width = max(self.status_width, len(text))
        print("\r" + padded, end="", file=self.fileobj, flush=True)

    def _clear_status_locked(self):
        if self.status_width <= 0:
            return
        print("\r" + (" " * self.status_width) + "\r", end="", file=self.fileobj, flush=True)
        self.status_width = 0

    def _thinking_loop(self, stop_event):
        while not stop_event.wait(self.tick_interval):
            started_at = self.thinking_started_at
            if started_at is None:
                return
            with self.lock:
                self.thinking_frame_index = (self.thinking_frame_index + 1) % len(THINKING_FRAMES)
                self._render_status(
                    _thinking_label(
                        self.time_fn() - started_at,
                        THINKING_FRAMES[self.thinking_frame_index],
                    )
                )

    def start_thinking(self):
        self.stop_thinking()
        self.thinking_started_at = self.time_fn()
        self.thinking_frame_index = 0
        with self.lock:
            self._render_status(_thinking_label(0.0, THINKING_FRAMES[self.thinking_frame_index]))
        stop_event = threading.Event()
        self._thinking_stop = stop_event
        self._thinking_thread = threading.Thread(
            target=self._thinking_loop,
            args=(stop_event,),
            daemon=True,
        )
        self._thinking_thread.start()

    def stop_thinking(self):
        # 打印普通行或最终答案增量前，必须先停止计时线程，
        # 否则 spinner 可能覆盖用户可见的正文。
        stop_event = self._thinking_stop
        thread = self._thinking_thread
        self._thinking_stop = None
        self._thinking_thread = None
        self.thinking_started_at = None
        if stop_event is not None:
            stop_event.set()
        if thread is not None:
            thread.join(timeout=0.2)
        with self.lock:
            self._clear_status_locked()

    def print_line(self, text):
        self.stop_thinking()
        with self.lock:
            print(text, file=self.fileobj, flush=True)

    def write_stream(self, text):
        if not text:
            return
        self.stop_thinking()
        with self.lock:
            print(text, end="", file=self.fileobj, flush=True)
        self.wrote = True

    def finish_stream(self, final_text):
        self.stop_thinking()
        with self.lock:
            if self.wrote:
                print("", file=self.fileobj, flush=True)
                return
            print(final_text, file=self.fileobj, flush=True)

    def close(self):
        self.stop_thinking()


class StreamPrinter:
    def __init__(self, ui):
        self.ui = ui

    def write(self, text):
        self.ui.write_stream(text)

    def finish(self, final_text):
        self.ui.finish_stream(final_text)


class StatusPrinter:
    def __init__(self, ui):
        self.ui = ui

    def emit(self, payload):
        event = str(payload.get("event", "")).strip()
        if event == "thinking":
            self.ui.start_thinking()
            return
        if event == "tool_started":
            name = str(payload.get("name", "")).strip() or "unknown"
            summary = _tool_arg_summary(payload.get("args", {}))
            suffix = f" {summary}" if summary else ""
            self.ui.print_line(f"Running tool: {name}{suffix}")
            return
        if event == "tool_finished":
            name = str(payload.get("name", "")).strip() or "unknown"
            duration_ms = payload.get("duration_ms")
            suffix = f" ({int(duration_ms)}ms)" if duration_ms is not None else ""
            self.ui.print_line(f"Tool finished: {name}{suffix}")

    def close(self):
        self.ui.close()


def _effective_model(args, provider):
    # 模型选择优先级：
    # 1. 用户显式传入 --model
    # 2. provider 对应的环境变量
    # 3. 代码里的默认值
    explicit_model = getattr(args, "model", None)
    if explicit_model:
        return explicit_model
    if provider == "openai":
        model = os.environ.get("OPENAI_MODEL")
        if model:
            return model
        return DEFAULT_OPENAI_MODEL
    if provider == "anthropic":
        model = os.environ.get("ANTHROPIC_MODEL")
        if model:
            return model
        return DEFAULT_ANTHROPIC_MODEL
    return DEFAULT_OLLAMA_MODEL


def _first_env(*names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _configured_secret_names(args):
    configured_secret_names = set(DEFAULT_SECRET_ENV_NAMES)
    configured_secret_names.update(str(name).upper() for name in args.secret_env_names)
    extra_names = os.environ.get(SECRET_ENV_NAMES_VAR, "")
    if not extra_names.strip():
        extra_names = os.environ.get(LEGACY_SECRET_ENV_NAMES_VAR, "")
    if extra_names.strip():
        configured_secret_names.update(
            item.strip().upper()
            for item in extra_names.split(",")
            if item.strip()
        )
    return sorted(configured_secret_names)


def _build_model_client(args):
    provider = getattr(args, "provider", "openai")
    # CLI 只负责把 provider 选择翻译成具体 client。
    # 真正的提示词格式、缓存支持、HTTP 协议差异，都封装在 models.py 里。
    if provider == "openai":
        model = _effective_model(args, provider)
        base_url = getattr(args, "base_url", None) or os.environ.get("OPENAI_API_BASE") or DEFAULT_OPENAI_BASE_URL
        api_key = os.environ.get("OPENAI_API_KEY", "")
        return OpenAICompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            timeout=getattr(args, "openai_timeout", getattr(args, "ollama_timeout", 300)),
        )
    if provider == "anthropic":
        model = _effective_model(args, provider)
        base_url = getattr(args, "base_url", None) or os.environ.get("ANTHROPIC_API_BASE") or DEFAULT_ANTHROPIC_BASE_URL
        api_key = _first_env("ANTHROPIC_API_KEY", "RIGHT_CODES_API_KEY", "OPENAI_API_KEY")
        return AnthropicCompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            timeout=getattr(args, "openai_timeout", getattr(args, "ollama_timeout", 300)),
        )

    model = _effective_model(args, provider)
    host = getattr(args, "host", DEFAULT_OLLAMA_HOST)
    return OllamaModelClient(
        model=model,
        host=host,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.ollama_timeout,
    )


def build_welcome(agent, model, host):
    width = max(68, min(shutil.get_terminal_size((80, 20)).columns, 84))
    inner = width - 4
    gap = 3
    left_width = (inner - gap) // 2
    right_width = inner - gap - left_width

    def row(text):
        body = middle(text, width - 4)
        return f"| {body.ljust(width - 4)} |"

    def divider(char="-"):
        return "+" + char * (width - 2) + "+"

    def center(text):
        body = middle(text, inner)
        return f"| {body.center(inner)} |"

    def cell(label, value, size):
        body = middle(f"{label:<9} {value}", size)
        return body.ljust(size)

    def pair(left_label, left_value, right_label, right_value):
        left = cell(left_label, left_value, left_width)
        right = cell(right_label, right_value, right_width)
        return f"| {left}{' ' * gap}{right} |"

    line = divider("=")
    rows = [center(text) for text in WELCOME_ART]
    rows.extend(
        [
            center(WELCOME_NAME),
            center(WELCOME_SUBTITLE),
            center(WELCOME_STATUS),
            divider("-"),
            row(""),
            row("WORKSPACE  " + middle(agent.workspace.cwd, inner - 11)),
            pair("MODEL", model, "BRANCH", agent.workspace.branch),
            pair("APPROVAL", agent.approval_policy, "SESSION", agent.session["id"]),
            row(""),
        ]
    )
    return "\n".join([line, *rows, line])


def build_agent(args):
    """根据 CLI 参数装配出一个可运行的 RepoFox 实例。

    为什么存在：
    命令行参数只是字符串和开关，runtime 需要的是已经装配好的对象图：
    model client、workspace snapshot、session store、secret 配置等。
    这个函数负责把“启动参数”翻译成“agent 运行现场”。

    输入 / 输出：
    - 输入：`argparse` 解析后的 `args`
    - 输出：一个新的 `RepoFox`，或一个从旧 session 恢复出来的 `RepoFox`

    在 agent 链路里的位置：
    它是整个程序启动链路里最靠近 runtime 的装配点。`main()` 先调它，
    得到 agent 后，后面无论是 one-shot 还是 REPL 模式，都会落到 `ask()`。
    """
    # 这里是 CLI 到 runtime 的装配点：
    # 先整理 secret 名单，再采集工作区快照，随后决定是恢复旧 session
    # 还是创建一个新的 RepoFox 实例。
    configured_secret_names = _configured_secret_names(args)
    workspace = WorkspaceContext.build(args.cwd)
    store = SessionStore(workspace.repo_root + "/.repofox/sessions")
    model = _build_model_client(args)
    session_id = args.resume
    if session_id == "latest":
        session_id = store.latest()
    if session_id:
        return RepoFox.from_session(
            model_client=model,
            workspace=workspace,
            session_store=store,
            session_id=session_id,
            approval_policy=args.approval,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            secret_env_names=configured_secret_names,
        )
    return RepoFox(
        model_client=model,
        workspace=workspace,
        session_store=store,
        approval_policy=args.approval,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        secret_env_names=configured_secret_names,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="RepoFox local coding agent for Ollama, OpenAI-compatible, or Anthropic-compatible models.",
    )
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt.")
    parser.add_argument("--cwd", default=".", help="Workspace directory.")
    parser.add_argument("--provider", choices=("ollama", "openai", "anthropic"), default="openai", help="Model backend to use.")
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override. Defaults to qwen3.5:4b for Ollama, OPENAI_MODEL for openai, and ANTHROPIC_MODEL for anthropic when set.",
    )
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Ollama server URL.")
    parser.add_argument("--base-url", default=None, help="Provider API base URL for openai or anthropic.")
    parser.add_argument("--ollama-timeout", type=int, default=300, help="Ollama request timeout in seconds.")
    parser.add_argument("--openai-timeout", type=int, default=300, help="OpenAI-compatible request timeout in seconds.")
    parser.add_argument("--resume", default=None, help="Session id to resume or 'latest'.")
    parser.add_argument("--approval", choices=("ask", "auto", "never"), default="ask", help="Approval policy for risky tools.")
    parser.add_argument(
        "--secret-env-name",
        dest="secret_env_names",
        action="append",
        default=[],
        help="Extra environment variable names to treat as secrets for trace/report redaction.",
    )
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum tool/model iterations per request.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum model output tokens per step.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature sent to Ollama.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling value sent to Ollama.")
    parser.add_argument("--no-stream", dest="stream_output", action="store_false", help="Disable streaming final answer output when supported by the model backend.")
    parser.add_argument("--serve", action="store_true", help="Run an HTTP Server-Sent Events API instead of the terminal REPL.")
    parser.add_argument("--serve-host", default="127.0.0.1", help="Host for --serve mode.")
    parser.add_argument("--serve-port", type=int, default=8765, help="Port for --serve mode.")
    parser.set_defaults(stream_output=True)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    agent = build_agent(args)

    model = getattr(agent.model_client, "model", getattr(args, "model", DEFAULT_OLLAMA_MODEL))
    host = getattr(agent.model_client, "host", getattr(agent.model_client, "base_url", getattr(args, "host", DEFAULT_OLLAMA_HOST)))

    if args.serve:
        return serve_agent(
            agent,
            host=args.serve_host,
            port=args.serve_port,
            stream_output=args.stream_output,
        )

    print(build_welcome(agent, model=model, host=host))

    if args.prompt:
        # one-shot 模式：只跑一次 ask，不进入 REPL 循环。
        prompt = " ".join(args.prompt).strip()
        if prompt:
            print()
            ui = ConsoleUI(sys.stdout)
            printer = StreamPrinter(ui)
            status_printer = StatusPrinter(ui)
            try:
                result = agent.ask(
                    prompt,
                    stream_final=printer.write if args.stream_output else None,
                    status_update=status_printer.emit,
                )
                printer.finish(result)
            except KeyboardInterrupt:
                print("\ninterrupted", file=sys.stderr)
                return 130
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 1
            finally:
                status_printer.close()
        return 0

    while True:
        # 交互模式：每次读取一条用户输入，交给同一个 agent，
        # 因此 session history 和 working memory 会跨轮延续。
        try:
            user_input = input("\nrepofox> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            return 0
        if user_input == "/help":
            print(HELP_DETAILS)
            continue
        if user_input == "/memory":
            print(agent.memory_text())
            continue
        if user_input == "/session":
            print(agent.session_path)
            continue
        if user_input == "/reset":
            agent.reset()
            print("session reset")
            continue

        print()
        ui = ConsoleUI(sys.stdout)
        printer = StreamPrinter(ui)
        status_printer = StatusPrinter(ui)
        try:
            result = agent.ask(
                user_input,
                stream_final=printer.write if args.stream_output else None,
                status_update=status_printer.emit,
            )
            printer.finish(result)
        except KeyboardInterrupt:
            print("\ninterrupted")
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
        finally:
            status_printer.close()
