"""Core server management logic (UI-agnostic)."""

from dataclasses import dataclass, field
import json
import subprocess
import threading
import webbrowser
from typing import Callable, Dict, List, Optional


@dataclass
class ServerConfig:
    """Plain config container (no tk variables)."""
    model_path: str = ""
    alias: str = ""
    lora_path: str = ""
    mmproj_path: str = ""
    chat_template: str = ""
    reasoning_format: str = ""
    reasoning_effort: str = ""
    jinja: bool = False

    ctx_size: int = 4096
    gpu_layers: int = 99
    threads: str = ""
    batch_size: str = ""
    ubatch_size: str = ""
    parallel: str = ""
    cont_batching: bool = False

    flash_attn: str = "auto"
    moe_cpu_layers: str = ""
    mlock: bool = False
    no_mmap: bool = False
    numa: bool = False
    cache_type_k: str = ""
    cache_type_v: str = ""

    draft_model_path: str = ""
    draft_gpu_layers: str = ""
    draft_tokens: str = ""

    host: str = "127.0.0.1"
    port: str = "8080"
    api_key: str = ""
    no_webui: bool = False
    embedding: bool = False
    verbose: bool = False

    n_predict: str = ""
    ignore_eos: bool = False
    temp: str = ""
    top_k: str = ""
    top_p: str = ""
    repeat_penalty: str = ""

    custom_arguments: List[Dict[str, object]] = field(default_factory=list)


@dataclass
class ServerProcess:
    """Runtime process holder."""
    process: Optional[subprocess.Popen] = None
    is_running: bool = False


class ServerManager:
    """Handles server command generation and process lifecycle."""
    def __init__(self):
        self.processes: Dict[str, ServerProcess] = {}

    def generate_command(self, config: ServerConfig) -> Optional[List[str]]:
        if not config.model_path.strip():
            return None

        cmd = ["llama-server", "-m", config.model_path.strip()]
        cmd.extend(["-c", str(config.ctx_size)])
        cmd.extend(["-ngl", str(config.gpu_layers)])

        args = {
            "--host": config.host,
            "--port": config.port,
            "-a": config.alias,
            "--api-key": config.api_key,
            "-t": config.threads,
            "-b": config.batch_size,
            "-np": config.parallel,
            "--lora": config.lora_path,
            "--mmproj": config.mmproj_path,
            "--chat-template": config.chat_template,
            "-md": config.draft_model_path,
            "-ngld": config.draft_gpu_layers,
            "--draft": config.draft_tokens,
            "--n-cpu-moe": config.moe_cpu_layers,
            "--reasoning-format": config.reasoning_format,
            "-ub": config.ubatch_size,
            "-n": config.n_predict,
            "--temp": config.temp,
            "--top-k": config.top_k,
            "--top-p": config.top_p,
            "--repeat-penalty": config.repeat_penalty,
            "--cache-type-k": config.cache_type_k,
            "--cache-type-v": config.cache_type_v,
        }
        for flag, value in args.items():
            if str(value).strip():
                cmd.extend([flag, str(value).strip()])

        if str(config.reasoning_effort).strip():
            kwargs_json = json.dumps({"reasoning_effort": config.reasoning_effort})
            cmd.extend(["--chat-template-kwargs", kwargs_json])

        if str(config.flash_attn).strip() and config.flash_attn != "auto":
            cmd.extend(["-fa", str(config.flash_attn).strip()])

        bool_args = {
            "--no-mmap": config.no_mmap,
            "--no-webui": config.no_webui,
            "-cb": config.cont_batching,
            "--mlock": config.mlock,
            "--embedding": config.embedding,
            "--jinja": config.jinja,
            "-v": config.verbose,
            "--ignore-eos": config.ignore_eos,
        }
        for flag, enabled in bool_args.items():
            if enabled:
                cmd.append(flag)

        if config.numa:
            cmd.extend(["--numa", "distribute"])

        for arg_item in config.custom_arguments:
            if arg_item.get("enabled", False) and str(arg_item.get("value", "")).strip():
                cmd.extend(str(arg_item["value"]).strip().split())

        return cmd

    def start_server(
        self,
        server_id: str,
        config: ServerConfig,
        on_output: Optional[Callable[[str], None]] = None,
        on_stopped: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> bool:
        if server_id not in self.processes:
            self.processes[server_id] = ServerProcess()

        state = self.processes[server_id]
        if state.is_running:
            return False

        cmd = self.generate_command(config)
        if not cmd:
            return False

        def emit(text: str):
            if on_output:
                on_output(text)

        def run_server():
            try:
                startupinfo = None
                if subprocess.os.name == "nt":
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

                state.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                    startupinfo=startupinfo,
                    encoding="utf-8",
                    errors="replace",
                )

                for line in iter(state.process.stdout.readline, ""):
                    emit(line)

                state.process.wait()
                state.is_running = False
                if on_stopped:
                    on_stopped()

            except FileNotFoundError:
                msg = "⚠ Error: 'llama-server' executable not found. Ensure it's in PATH or app directory."
                if on_error:
                    on_error(msg)
                state.is_running = False
                if on_stopped:
                    on_stopped()
            except Exception as e:
                if on_error:
                    on_error(f"⚠ Error starting server: {e}")
                state.is_running = False
                if on_stopped:
                    on_stopped()

        state.is_running = True
        threading.Thread(target=run_server, daemon=True).start()
        return True

    def stop_server(self, server_id: str) -> bool:
        state = self.processes.get(server_id)
        if not state or not state.is_running or not state.process:
            return False
        try:
            state.process.terminate()
            return True
        except Exception:
            return False

    def any_server_running(self) -> bool:
        return any(state.is_running for state in self.processes.values())

    def terminate_all_servers(self):
        for state in self.processes.values():
            if state.process and state.is_running:
                try:
                    state.process.terminate()
                except Exception:
                    pass
                state.is_running = False

    def open_browser(self, host: str, port: str) -> bool:
        host = host.strip()
        if host == "0.0.0.0":
            host = "localhost"
        url = f"http://{host}:{port.strip()}"
        try:
            webbrowser.open(url)
            return True
        except Exception:
            return False