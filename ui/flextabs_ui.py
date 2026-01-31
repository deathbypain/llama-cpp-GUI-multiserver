"""Flextabs-based UI implementation using core modules."""

import tkinter as tk
from tkinter import filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
from ttkbootstrap.widgets.scrolled import ScrolledText, ScrolledFrame
from ttkbootstrap.widgets import ToolTip
from flextabs import TabManager, TabConfig, TabContent
from dataclasses import asdict

from core import ServerManager, ServerConfig, ConfigHandler, TrayManager, browse_file


class TabContentExtension(TabContent):
    """Extension of flextabs.TabContent to add common functionality."""
    def create_button(self, parent, text, command, tooltip_text, state=tk.NORMAL, bootstyle="primary"):
        btn = ttk.Button(parent, text=text, command=command, state=state, bootstyle=bootstyle)
        btn.pack(side=tk.LEFT, padx=(0, 5))
        ToolTip(btn, text=tooltip_text)
        return btn


class HomeTabContent(TabContentExtension):
    def setup_content(self):
        text = tk.Text(
            self.frame,
            wrap="word",
            font=("Segoe UI", 11),
            background="white",
            foreground="black",
        )
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        intro_text = (
            "Welcome to the LLaMA Server GUI!\n\n"
            "QUICK START\n"
            "• Load a Model: Use the \"Browse\" button in the Models tab to select your GGUF model file\n"
            "• Configure Settings: Adjust parameters across the different tabs as needed\n"
            "• Start Server: Click the \"Start Server\" button\n"
            "• Access Web UI: Click \"Open Browser\" to access the llama.cpp web interface\n\n"
            "CONFIGURATION TABS\n"
            "📁 Models\n"
            "  • Primary Model: Select your main GGUF model file\n"
            "  • Model Extensions: Add LoRA adapters or multimodal projectors\n"
            "  • Chat Behavior: Configure chat templates and reasoning settings\n\n"
            "⚙️ Generation\n"
            "  • Output Control: Set token limits and generation behavior\n"
            "  • Sampling Parameters: Fine-tune creativity and randomness\n\n"
            "🚀 Performance\n"
            "  • Core Performance: Context size, GPU layers, CPU threads\n"
            "  • Advanced Throughput: Parallel processing and continuous batching\n\n"
            "🔬 Advanced\n"
            "  • Memory Optimizations: Flash attention, memory locking, NUMA settings\n"
            "  • Speculative Decoding: Use draft models for faster inference\n\n"
            "🌐 Server & API\n"
            "  • Network Configuration: Host, port, and API key settings\n"
            "  • Custom Arguments: Add any additional llama-server flags\n"
            "  • Access Control: Configure web UI and API access\n\n"
            "📊 Server Output\n"
            "  • Live Monitoring: Real-time server log output\n"
            "  • Log Management: Clear output and monitor server status\n\n"
            "CONFIGURATION MANAGEMENT\n"
            "• Save Config: Preserve your current settings to a JSON config\n"
            "• Load Config: Restore previously saved configurations\n"
            "• Portable: Config file is stored in the application directory\n\n"
            "SYSTEM TRAY (Optional)\n"
            "When pystray is installed, the application can minimize to the system tray:\n"
            "• Minimize to Tray: Close the window to hide in the system tray\n"
            "• Tray Menu: Right-click the tray icon for quick actions\n"
            "• Background Operation: Keep the server running while GUI is hidden"
        )
        scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.config(yscrollcommand=scrollbar.set)

        text.insert(tk.END, intro_text)
        text.config(state=tk.DISABLED)


class SettingsTabContent(TabContentExtension):
    def setup_content(self):
        ttk.Label(self.frame, text="Settings Configuration").pack(pady=20)
        self.manager().add_close_button(self.frame, self.tab_id).pack(pady=10)


class ServerTabState:
    """Holds per-tab server configuration, UI references, and runtime state."""
    def __init__(self, master):
        self.server_id = None
        self.is_running = False

        self.start_button = None
        self.stop_button = None
        self.browser_button = None
        self.output_text = None
        self.new_arg_entry = None
        self.custom_args_list_frame = None
        self.notebook = None                # to store the notebook reference

        self.custom_arguments = []
        self.slider_refs = {}

        self.model_path = tk.StringVar(master=master)
        self.alias = tk.StringVar(master=master)
        self.lora_path = tk.StringVar(master=master)
        self.mmproj_path = tk.StringVar(master=master)
        self.chat_template = tk.StringVar(master=master)
        self.reasoning_format = tk.StringVar(master=master)
        self.reasoning_effort = tk.StringVar(master=master)
        self.jinja = tk.BooleanVar(master=master, value=False)

        self.n_predict = tk.StringVar(master=master, value="")
        self.ignore_eos = tk.BooleanVar(master=master, value=False)
        self.temp = tk.StringVar(master=master, value="")
        self.top_k = tk.StringVar(master=master, value="")
        self.top_p = tk.StringVar(master=master, value="")
        self.repeat_penalty = tk.StringVar(master=master, value="")

        self.ctx_size = tk.IntVar(master=master, value=4096)
        self.gpu_layers = tk.IntVar(master=master, value=99)
        self.threads = tk.StringVar(master=master, value="")
        self.batch_size = tk.StringVar(master=master, value="")
        self.ubatch_size = tk.StringVar(master=master, value="")
        self.parallel = tk.StringVar(master=master, value="")
        self.cont_batching = tk.BooleanVar(master=master, value=False)

        self.flash_attn = tk.StringVar(master=master, value="auto")
        self.moe_cpu_layers = tk.StringVar(master=master, value="")
        self.mlock = tk.BooleanVar(master=master, value=False)
        self.no_mmap = tk.BooleanVar(master=master, value=False)
        self.numa = tk.BooleanVar(master=master, value=False)
        self.cache_type_k = tk.StringVar(master=master, value="")
        self.cache_type_v = tk.StringVar(master=master, value="")

        self.draft_model_path = tk.StringVar(master=master)
        self.draft_gpu_layers = tk.StringVar(master=master, value="")
        self.draft_tokens = tk.StringVar(master=master, value="")

        self.host = tk.StringVar(master=master, value="127.0.0.1")
        self.port = tk.StringVar(master=master, value="8080")
        self.api_key = tk.StringVar(master=master)
        self.no_webui = tk.BooleanVar(master=master, value=False)
        self.embedding = tk.BooleanVar(master=master, value=False)
        self.verbose = tk.BooleanVar(master=master, value=False)


class ServerTabContent(TabContentExtension):
    server_manager: ServerManager = None
    app = None

    def setup_content(self):
        self.root = self.frame.winfo_toplevel()
        state = ServerTabState(self.frame)
        state.server_id = self.tab_id
        self.app.register_tab_state(self.tab_id, state)

        control_frame = ttk.Frame(self.frame)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

        left_button_frame = ttk.Frame(control_frame)
        left_button_frame.pack(side=tk.LEFT)

        self.create_button(left_button_frame, "Save As 💾", lambda s=state: self.save_config(state=s),
                           "Save the current settings to a chosen file.", bootstyle="secondary")
        self.create_button(left_button_frame, "Load (Browse) 📂", lambda s=state: self.load_config(state=s, browse=True),
                           "Browse and load a saved config.", bootstyle="secondary")
        self.create_button(left_button_frame, "Generate Command ⚡", lambda s=state: self.show_command(state=s),
                           "Show the final command to be executed.", bootstyle="info")

        right_button_frame = ttk.Frame(control_frame)
        right_button_frame.pack(side=tk.RIGHT)
        state.browser_button = self.create_button(right_button_frame, "Open Browser 🌐", lambda s=state: self.open_browser(state=s),
                                "Access the server web UI.", state=tk.DISABLED,
                                bootstyle="primary-outline")
        state.stop_button = self.create_button(right_button_frame, "Stop Server ⏹️", lambda s=state: self.stop_server(state=s),
                                "Stop the running server process.", state=tk.DISABLED,
                                bootstyle="danger")
        state.start_button = self.create_button(right_button_frame, "Start Server ▶️", lambda s=state: self.start_server(state=s),
                                "Start the server with current settings.", bootstyle="success")
        
        notebook = ttk.Notebook(self.frame, bootstyle="primary")
        notebook.pack(fill=tk.BOTH, expand=True)
        state.notebook = notebook  # Store the notebook reference in state

        model_frame = ttk.Frame(notebook, padding="10")
        generation_frame = ttk.Frame(notebook, padding="10")
        performance_core_frame = ttk.Frame(notebook, padding="10")
        performance_advanced_frame = ttk.Frame(notebook, padding="10")
        server_api_frame = ttk.Frame(notebook, padding="10")
        output_frame = ttk.Frame(notebook, padding="10")

        notebook.add(model_frame, text="  Models ")
        notebook.add(generation_frame, text=" Generation ")
        notebook.add(performance_core_frame, text=" Performance ")
        notebook.add(performance_advanced_frame, text=" Advanced ")
        notebook.add(server_api_frame, text=" Server & API ")
        notebook.add(output_frame, text=" Server Output ")

        self.setup_model_tab(model_frame, state)
        self.setup_generation_tab(generation_frame, state)
        self.setup_performance_core_tab(performance_core_frame, state)
        self.setup_performance_advanced_tab(performance_advanced_frame, state)
        self.setup_server_api_tab(server_api_frame, state)
        self.setup_output_tab(output_frame, state)

        return state

    def build_config(self, state: ServerTabState) -> ServerConfig:
        return ServerConfig(
            model_path=state.model_path.get(),
            alias=state.alias.get(),
            lora_path=state.lora_path.get(),
            mmproj_path=state.mmproj_path.get(),
            chat_template=state.chat_template.get(),
            reasoning_format=state.reasoning_format.get(),
            reasoning_effort=state.reasoning_effort.get(),
            jinja=state.jinja.get(),

            ctx_size=state.ctx_size.get(),
            gpu_layers=state.gpu_layers.get(),
            threads=state.threads.get(),
            batch_size=state.batch_size.get(),
            ubatch_size=state.ubatch_size.get(),
            parallel=state.parallel.get(),
            cont_batching=state.cont_batching.get(),

            flash_attn=state.flash_attn.get(),
            moe_cpu_layers=state.moe_cpu_layers.get(),
            mlock=state.mlock.get(),
            no_mmap=state.no_mmap.get(),
            numa=state.numa.get(),
            cache_type_k=state.cache_type_k.get(),
            cache_type_v=state.cache_type_v.get(),

            draft_model_path=state.draft_model_path.get(),
            draft_gpu_layers=state.draft_gpu_layers.get(),
            draft_tokens=state.draft_tokens.get(),

            host=state.host.get(),
            port=state.port.get(),
            api_key=state.api_key.get(),
            no_webui=state.no_webui.get(),
            embedding=state.embedding.get(),
            verbose=state.verbose.get(),

            n_predict=state.n_predict.get(),
            ignore_eos=state.ignore_eos.get(),
            temp=state.temp.get(),
            top_k=state.top_k.get(),
            top_p=state.top_p.get(),
            repeat_penalty=state.repeat_penalty.get(),

            custom_arguments=state.custom_arguments,
        )

    def apply_config_dict(self, state: ServerTabState, config: dict):
        state.model_path.set(config.get('model_path', ''))
        state.alias.set(config.get('alias', ''))
        state.lora_path.set(config.get('lora_path', ''))
        state.mmproj_path.set(config.get('mmproj_path', ''))
        state.chat_template.set(config.get('chat_template', ''))
        state.reasoning_effort.set(config.get('reasoning_effort', ''))
        state.jinja.set(config.get('jinja', False))

        state.ctx_size.set(config.get('ctx_size', 4096))
        state.gpu_layers.set(config.get('gpu_layers', 99))
        state.threads.set(config.get('threads', ''))
        state.batch_size.set(config.get('batch_size', ''))
        state.cont_batching.set(config.get('cont_batching', False))
        state.parallel.set(config.get('parallel', ''))
        state.flash_attn.set(config.get('flash_attn', False))
        state.mlock.set(config.get('mlock', False))
        state.no_mmap.set(config.get('no_mmap', False))
        state.numa.set(config.get('numa', False))
        state.moe_cpu_layers.set(config.get('moe_cpu_layers', ''))
        state.draft_model_path.set(config.get('draft_model_path', ''))
        state.draft_gpu_layers.set(config.get('draft_gpu_layers', ''))
        state.draft_tokens.set(config.get('draft_tokens', ''))
        state.host.set(config.get('host', '127.0.0.1'))
        state.port.set(config.get('port', '8080'))
        state.api_key.set(config.get('api_key', ''))
        state.no_webui.set(config.get('no_webui', False))
        state.embedding.set(config.get('embedding', False))
        state.verbose.set(config.get('verbose', False))

        state.custom_arguments = config.get('custom_arguments_list', [])
        if not state.custom_arguments and 'custom_args' in config:
            old_args_str = config['custom_args'].strip()
            if old_args_str:
                state.custom_arguments.append({"value": old_args_str, "enabled": True})
        self.rebuild_custom_args_list(state)

        state.reasoning_format.set(config.get('reasoning_format', ''))
        state.ubatch_size.set(config.get('ubatch_size', ''))
        state.n_predict.set(config.get('n_predict', ''))
        state.ignore_eos.set(config.get('ignore_eos', False))
        state.temp.set(config.get('temp', ''))
        state.top_k.set(config.get('top_k', ''))
        state.top_p.set(config.get('top_p', ''))
        state.repeat_penalty.set(config.get('repeat_penalty', ''))
        state.cache_type_k.set(config.get('cache_type_k', ''))
        state.cache_type_v.set(config.get('cache_type_v', ''))

        self.update_all_sliders(state)

    def show_command(self, state):
        config = self.build_config(state)
        cmd = self.server_manager.generate_command(config)
        if not cmd:
            Messagebox.show_error("Model path is required!", "Error")
            return
        command_str = " ".join(f'"{arg}"' if " " in arg else arg for arg in cmd)
        cmd_window = ttk.Toplevel(self.frame)
        cmd_window.title("Generated Command")
        cmd_window.geometry("1200x300")
        ttk.Label(cmd_window, text="Generated Command:", padding="10 10 0 5").pack(anchor=tk.W)
        cmd_text = ScrolledText(cmd_window, height=5, wrap=tk.WORD, autohide=True)
        cmd_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        cmd_text.insert(tk.END, command_str)
        cmd_text.text.configure(state=tk.DISABLED)

        def copy_command():
            cmd_window.clipboard_clear()
            cmd_window.clipboard_append(command_str)
            Messagebox.ok("Command copied to clipboard!", "Copied", parent=cmd_window)
        ttk.Button(cmd_window, text="Copy to Clipboard", command=copy_command).pack(pady=10)

    def start_server(self, state):
        if state.is_running:
            return
        config = self.build_config(state)
        cmd = self.server_manager.generate_command(config)
        if not cmd:
            Messagebox.show_error("Model path is required!", "Error")
            return

        state.output_text.delete(1.0, tk.END)
        command_str = " ".join(f'"{arg}"' if " " in arg else arg for arg in cmd)
        self.update_output(state, f"▶ Starting server with command:\n{command_str}\n\n" + "="*80 + "\n")

        def on_output(text):
            self.root.after(0, self.update_output, state, text)

        def on_stopped():
            self.root.after(0, self.server_stopped, state)

        def on_error(msg):
            self.root.after(0, self.update_output, state, f"\n{msg}\n")

        started = self.server_manager.start_server(
            state.server_id,
            config,
            on_output=on_output,
            on_stopped=on_stopped,
            on_error=on_error,
        )

        if started:
            state.is_running = True
            state.start_button.config(state=tk.DISABLED)
            state.stop_button.config(state=tk.NORMAL)
            state.browser_button.config(state=tk.NORMAL)
            
             # Auto-switch to Server Output tab (index 5)
            state.notebook.select(5)

    def stop_server(self, state):
        stopped = self.server_manager.stop_server(state.server_id)
        if stopped:
            self.update_output(state, "\n" + "="*80 + "\n⏹️ Server stop requested...\n")

    def server_stopped(self, state):
        state.is_running = False
        state.start_button.config(state=tk.NORMAL)
        state.stop_button.config(state=tk.DISABLED)
        state.browser_button.config(state=tk.DISABLED)
        self.update_output(state, "⏹️ Server process has terminated.\n")

    def update_output(self, state, text):
        state.output_text.insert(tk.END, text)
        state.output_text.see(tk.END)

    def clear_output(self, state):
        state.output_text.delete(1.0, tk.END)

    def save_config(self, state):
        config = asdict(self.build_config(state))
        config['custom_arguments_list'] = config.pop('custom_arguments', [])

        app_dir = self.app.get_app_dir()
        configs_dir = self.app.ensure_configs_dir(app_dir)
        save_path = filedialog.asksaveasfilename(
            title="Save Configuration As",
            defaultextension='.json',
            filetypes=[('JSON files', '*.json'), ('All files', '*.*')],
            initialdir=configs_dir,
            initialfile='config-'
        )
        if not save_path:
            return

        saved = ConfigHandler.save_server_config(config, filepath=save_path)
        if saved:
            Messagebox.ok(f"Configuration saved to {saved}", "Success")

    def load_config(self, state, browse=False):
        if not browse:
            return

        app_dir = self.app.get_app_dir()
        configs_dir = self.app.ensure_configs_dir(app_dir)
        chosen = filedialog.askopenfilename(
            title="Select Configuration",
            filetypes=[('JSON files', '*.json'), ('All files', '*.*')],
            initialdir=configs_dir
        )
        if not chosen:
            return

        config = ConfigHandler.load_server_config(chosen)
        if config is None:
            Messagebox.show_warning(f"Config file not found: {chosen}", "Not Found")
            return

        self.apply_config_dict(state, config)

    def open_browser(self, state):
        config = self.build_config(state)
        success = self.server_manager.open_browser(config.host, config.port)
        if success:
            self.update_output(state, f"🌐 Opened browser at http://{config.host}:{config.port}\n")

    # --- Tab UI setup helpers ---

    def setup_model_tab(self, parent, state):
        """Configures the 'Model' tab for model files, extensions, and chat behavior."""
        model_group = ttk.Labelframe(parent, text="Primary Model", padding="10")
        model_group.pack(fill=tk.X, pady=5)
        self.create_file_entry(model_group, "Model Path (-m):", state.model_path, "Path to the GGUF model file.", ".gguf", row=0)
        self.create_entry(model_group, "Model Alias (-a):", state.alias, "Set an alias for the model (used in API calls).", row=1)

        ext_group = ttk.Labelframe(parent, text="Model Extensions", padding="10")
        ext_group.pack(fill=tk.X, pady=5)
        self.create_file_entry(ext_group, "LoRA Path (--lora):", state.lora_path, "Path to a LoRA adapter file (optional).", ".gguf", row=0)
        self.create_file_entry(ext_group, "Multimodal Projector (--mmproj):", state.mmproj_path, "Path to a multimodal projector file (for vision models).", ".gguf", row=1)

        chat_group = ttk.Labelframe(parent, text="Chat Behavior", padding="10")
        chat_group.pack(fill=tk.X, pady=5)
        chat_templates = ["", "bailing", "chatglm3", "chatglm4", "chatml", "command-r", "deepseek", "deepseek2", "gemma", "llama2", "llama3", "mistral", "openchat", "phi3", "vicuna", "zephyr"]
        self.create_combobox(chat_group, "Template (--chat-template):", state.chat_template, "Select a chat template (leave blank for auto-detection).", chat_templates, row=0)

        reasoning_formats = ["", "auto", "none", "deepseek"]
        self.create_combobox(chat_group, "Reasoning Format (--reasoning-format):", state.reasoning_format, "Controls whether thought tags are allowed and/or extracted from the response.", reasoning_formats, row=1)

        reasoning_levels = ["", "low", "medium", "high"]
        self.create_combobox(chat_group, "Reasoning Effort:", state.reasoning_effort, "Set reasoning effort for chat template kwargs (some models).", reasoning_levels, row=2)

        self.create_checkbutton(chat_group, "Enable Jinja (--jinja)", state.jinja, "Enable Jinja2 templating (required for some custom templates).", row=3)

    def setup_generation_tab(self, parent, state):
        """Configures the 'Generation' tab for sampling and output control."""
        output_group = ttk.Labelframe(parent, text="Output Control", padding="10")
        output_group.pack(fill=tk.X, pady=5, side=tk.TOP)

        self.create_spinbox(output_group, "Tokens to Generate (-n, --n-predict):", state.n_predict, "Number of tokens to generate (default -1 = infinite).", from_=-1, to=131072, increment=1, row=0)
        self.create_checkbutton(output_group, "Ignore End-of-Sequence (--ignore-eos)", state.ignore_eos, "Prevents model from stopping early.", row=1)

        sampling_group = ttk.Labelframe(parent, text="Sampling Parameters", padding="10")
        sampling_group.pack(fill=tk.X, pady=5)

        self.create_spinbox(sampling_group, "Temperature (--temp):", state.temp, "Creativity level (default 0.8). Lower = deterministic, higher = creative.", from_=0, to=2, increment=0.1, row=0)
        self.create_spinbox(sampling_group, "Top-K (--top-k):", state.top_k, "Keep only top-k tokens when sampling (default 40).", from_=0, to=1000, increment=1, row=1)
        self.create_spinbox(sampling_group, "Top-P (--top-p):", state.top_p, "Nucleus sampling (default 0.9).", from_=0, to=1, increment=0.1, row=2)
        self.create_spinbox(sampling_group, "Repeat Penalty (--repeat-penalty):", state.repeat_penalty, "Penalizes repetition (default 1.0). Increase to reduce loops.", from_=0, to=2, increment=0.1, row=3)

    def setup_performance_core_tab(self, parent, state):
        """Configures the 'Performance' tab for core speed and throughput settings."""
        core_group = ttk.Labelframe(parent, text="Core Performance", padding="10")
        core_group.pack(fill=tk.X, pady=5, side=tk.TOP)
        self.create_slider(core_group, "Context Size (-c):", state.ctx_size, "Context size (sequence length) for the model.", from_=0, to=131072, resolution=1024, row=0, state=state)
        self.create_slider(core_group, "GPU Layers (-ngl):", state.gpu_layers, "Number of model layers to offload to GPU (99 for all).", from_=0, to=99, resolution=1, row=1, state=state)
        self.create_spinbox(core_group, "CPU Threads (-t):", state.threads, "Number of CPU threads to use (e.g., 8).", from_=1, to=128, increment=1, row=2)
        self.create_spinbox(core_group, "Batch Size (-b):", state.batch_size, "Batch size for prompt processing (e.g., 2048).", from_=1, to=8192, increment=1, row=3)
        self.create_spinbox(core_group, "Physical Batch Size (-ub):", state.ubatch_size, "Physical batch size. Lower values reduce VRAM use but slow things down.", from_=1, to=1024, increment=1, row=4)

        throughput_group = ttk.Labelframe(parent, text="Advanced Throughput", padding="10")
        throughput_group.pack(fill=tk.X, pady=5)
        self.create_spinbox(throughput_group, "Parallel Sequences (-np):", state.parallel, "Number of parallel sequences to process (e.g., 4).", row=0, from_=1, to=16, increment=1)
        self.create_checkbutton(throughput_group, "Continuous Batching (-cb)", state.cont_batching, "Enable continuous batching for higher throughput.", row=1)

    def setup_performance_advanced_tab(self, parent, state):
        """Configures the 'Advanced' tab for memory, optimizations, and speculative decoding."""
        mem_group = ttk.Labelframe(parent, text="Memory & Optimizations", padding="10")
        mem_group.pack(fill=tk.X, pady=5)
        flash_attn_options = ["on", "off", "auto"]
        self.create_combobox(mem_group, "Flash Attention (-fa):", state.flash_attn, "Set Flash Attention use ('on', 'off', or 'auto', default: 'auto').", flash_attn_options, row=0)
        self.create_spinbox(mem_group, "MoE CPU Layers (--n-cpu-moe):", state.moe_cpu_layers, "MoE layers to keep on CPU if model doesn't fit on GPU.", row=1, from_=0, to=99, increment=1)
        self.create_checkbutton(mem_group, "Memory Lock (--mlock)", state.mlock, "Lock model in RAM to prevent swapping.", row=2)
        self.create_checkbutton(mem_group, "No Memory Mapping (--no-mmap)", state.no_mmap, "Disable memory mapping of the model file.", row=3)
        self.create_checkbutton(mem_group, "NUMA Optimizations (--numa)", state.numa, "Enable NUMA-aware optimizations for specific hardware.", row=4)
        cache_types = ["", "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
        self.create_combobox(mem_group, "Cache Type K (-ctk, --cache-type-k):", state.cache_type_k, "KV cache data type for K (default: f16).", cache_types, row=5)
        self.create_combobox(mem_group, "Cache Type V (-ctv, --cache-type-v):", state.cache_type_v, "KV cache data type for V (default: f16).", cache_types, row=6)

        spec_group = ttk.Labelframe(parent, text="Speculative Decoding", padding="10")
        spec_group.pack(fill=tk.X, pady=5)
        self.create_file_entry(spec_group, "Draft Model (-md):", state.draft_model_path, "Path to the draft model for speculative decoding.", ".gguf", row=0)
        self.create_spinbox(spec_group, "Draft GPU Layers (-ngld):", state.draft_gpu_layers, "Number of GPU layers for the draft model.", row=1, from_=0, to=99, increment=1)
        self.create_spinbox(spec_group, "Draft Tokens (--draft):", state.draft_tokens, "Number of tokens to draft (e.g., 5).", row=2, from_=1, to=1024, increment=1)

    def setup_server_api_tab(self, parent, state):
        """Configures the 'Server & API' tab for network, access, and logging."""
        parent.rowconfigure(2, weight=1)
        parent.columnconfigure(0, weight=1)

        net_group = ttk.Labelframe(parent, text="Network Configuration", padding="10")
        net_group.grid(row=0, column=0, sticky=EW, pady=5)
        net_group.columnconfigure(1, weight=1)
        self.create_entry(net_group, "Host (--host):", state.host, "IP address to listen on (0.0.0.0 for network access).", row=0)
        self.create_entry(net_group, "Port (--port):", state.port, "Network port for the server to listen on.", row=1)

        access_group = ttk.Labelframe(parent, text="Access & Features", padding="10")
        access_group.grid(row=1, column=0, sticky=EW, pady=5)
        access_group.columnconfigure(1, weight=1)
        self.create_entry(access_group, "API Key (--api-key):", state.api_key, "API key for bearer token authentication (optional).", row=0)
        self.create_checkbutton(access_group, "Disable Web UI (--no-webui)", state.no_webui, "Disable the built-in web interface.", row=1)
        self.create_checkbutton(access_group, "Embeddings Only (--embedding)", state.embedding, "Enable embedding-only mode (disables chat).", row=2)

        custom_group = ttk.Labelframe(parent, text="Custom Arguments Management", padding="10")
        custom_group.grid(row=2, column=0, sticky=NSEW, pady=5)
        custom_group.columnconfigure(0, weight=1)
        custom_group.rowconfigure(1, weight=1)

        add_arg_frame = ttk.Frame(custom_group)
        add_arg_frame.grid(row=0, column=0, sticky=EW, pady=(0, 10))
        add_arg_frame.columnconfigure(0, weight=1)
        state.new_arg_entry = ttk.Entry(add_arg_frame)
        state.new_arg_entry.grid(row=0, column=0, sticky=EW, padx=(0, 5))
        ToolTip(state.new_arg_entry, "Enter a full argument with its value (e.g., --my-flag value) and press Add.")
        add_button = ttk.Button(add_arg_frame, text="Add", command=lambda s=state: self.add_custom_argument(s), bootstyle="success-outline")
        add_button.grid(row=0, column=1, sticky=E)

        state.custom_args_list_frame = ScrolledFrame(custom_group, autohide=True, bootstyle="round")
        state.custom_args_list_frame.grid(row=1, column=0, sticky=NSEW)

        other_options_frame = ttk.Frame(custom_group)
        other_options_frame.grid(row=2, column=0, sticky=EW, pady=(10, 0))
        verbose_cb = ttk.Checkbutton(other_options_frame, text="Verbose Logging (-v)", variable=state.verbose, bootstyle="round-toggle")
        verbose_cb.pack(side=tk.LEFT)
        ToolTip(verbose_cb, "Enable verbose server logging for debugging.")

    def setup_output_tab(self, parent, state):
        """Sets up the server output log view."""
        ttk.Label(parent, text="Server Log Output:").pack(anchor=tk.W, pady=(0, 5))
        monospace_font = ("Consolas", 10)
        state.output_text = ScrolledText(parent, height=20, wrap=tk.WORD, font=monospace_font, autohide=True)
        state.output_text.pack(fill=tk.BOTH, expand=True)
        clear_btn = ttk.Button(parent, text="Clear Output", command=lambda s=state: self.clear_output(state=s), bootstyle="secondary-outline")
        clear_btn.pack(pady=(10, 0), anchor=tk.E)
        ToolTip(clear_btn, "Clear all text from the log output window.")

    # --- UI Helper Methods ---
    def create_file_entry(self, parent, label_text, string_var, tooltip_text, file_ext, row):
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        file_path_frame = ttk.Frame(parent)
        file_path_frame.grid(row=row, column=1, sticky=tk.EW, pady=5)
        parent.columnconfigure(1, weight=1)
        entry = ttk.Entry(file_path_frame, textvariable=string_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        browse_btn = ttk.Button(
            file_path_frame,
            text="Browse",
            command=lambda: browse_file(string_var, file_ext),  # Use the imported function
            bootstyle="primary"
        )
        browse_btn.pack(side=tk.RIGHT)
        ToolTip(label, text=tooltip_text)
        ToolTip(entry, text=tooltip_text)
        ToolTip(browse_btn, text=f"Select a {file_ext} file.")

    def create_entry(self, parent, label_text, string_var, tooltip_text, row):
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent, textvariable=string_var, width=30)
        entry.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=5)
        parent.columnconfigure(1, weight=1)
        ToolTip(label, text=tooltip_text)
        ToolTip(entry, text=tooltip_text)

    def create_spinbox(self, parent, label_text, variable, tooltip_text, from_, to, increment, row):
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)

        spin = ttk.Spinbox(
            parent,
            textvariable=variable,
            from_=from_,
            to=to,
            increment=increment,
            width=10,
            bootstyle="primary"
        )
        spin.grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)

        ToolTip(label, text=tooltip_text)
        ToolTip(spin, text=tooltip_text)
        return spin

    def create_combobox(self, parent, label_text, string_var, tooltip_text, values, row):
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        combobox = ttk.Combobox(parent, textvariable=string_var, values=values)
        combobox.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=5)
        parent.columnconfigure(1, weight=1)
        ToolTip(label, text=tooltip_text)
        ToolTip(combobox, text=tooltip_text)

    def create_slider(self, parent, label_text, int_var, tooltip_text, from_, to, resolution, row, state):
        slider_frame = ttk.Frame(parent)
        slider_frame.grid(row=row, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        parent.columnconfigure(1, weight=1)
        label = ttk.Label(slider_frame, text=label_text)
        label.pack(anchor=tk.W)
        ToolTip(label, text=tooltip_text)
        control_frame = ttk.Frame(slider_frame)
        control_frame.pack(fill=tk.X, pady=(2, 0))
        slider = ttk.Scale(
            control_frame,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            variable=int_var,
            command=lambda v: self.update_slider_label(int_var, value_label, resolution),
            bootstyle="primary"
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        ToolTip(slider, text=tooltip_text)
        value_label = ttk.Label(control_frame, text=str(int_var.get()), width=8, anchor=tk.CENTER)
        value_label.pack(side=tk.RIGHT)

        slider_key = f"{label_text}_{id(int_var)}"
        state.slider_refs[slider_key] = {
            'var': int_var,
            'slider': slider,
            'label': value_label,
            'resolution': resolution
        }
        self.update_slider_label(int_var, value_label, resolution)

    def update_slider_label(self, int_var, label, resolution):
        raw_value = int_var.get()
        rounded_value = round(raw_value / resolution) * resolution
        int_var.set(rounded_value)
        label.config(text=str(rounded_value))

    def update_all_sliders(self, state):
        for key, refs in state.slider_refs.items():
            refs['slider'].set(refs['var'].get())
            self.update_slider_label(refs['var'], refs['label'], refs['resolution'])

    def create_checkbutton(self, parent, text, variable, tooltip_text, row):
        cb = ttk.Checkbutton(parent, text=text, variable=variable, bootstyle="round-toggle")
        cb.grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        ToolTip(cb, text=tooltip_text)

    # --- Custom Argument Methods ---
    def add_custom_argument(self, state):
        arg_text = state.new_arg_entry.get().strip()
        if not arg_text:
            return
        if any(arg['value'] == arg_text for arg in state.custom_arguments):
            Messagebox.show_warning("This argument already exists in the list.", "Duplicate Argument")
            return

        state.custom_arguments.append({"value": arg_text, "enabled": True})
        state.new_arg_entry.delete(0, tk.END)
        self.rebuild_custom_args_list(state)

    def delete_custom_argument(self, state, arg_to_delete):
        state.custom_arguments.remove(arg_to_delete)
        self.rebuild_custom_args_list(state)

    def rebuild_custom_args_list(self, state):
        for widget in state.custom_args_list_frame.winfo_children():
            widget.destroy()

        for arg_item in state.custom_arguments:
            row_frame = ttk.Frame(state.custom_args_list_frame, padding=(5, 3))
            row_frame.pack(fill=X, expand=True, padx=(0, 5))

            is_enabled_var = tk.BooleanVar(value=arg_item.get("enabled", True))

            def on_toggle(item=arg_item, var=is_enabled_var):
                item["enabled"] = var.get()

            toggle = ttk.Checkbutton(row_frame, variable=is_enabled_var, bootstyle="round-toggle", command=on_toggle)
            toggle.pack(side=LEFT, padx=(0, 10))

            label = ttk.Label(row_frame, text=arg_item["value"])
            delete_btn = ttk.Button(row_frame, text="Delete", bootstyle="danger-link",
                                    command=lambda item=arg_item: self.delete_custom_argument(state, item))

            delete_btn.pack(side=RIGHT, padx=(10, 0))

            def start_edit(event, item, lbl, frame, del_btn):
                lbl.pack_forget()

                entry_var = tk.StringVar(value=item["value"])
                edit_entry = ttk.Entry(frame, textvariable=entry_var)
                edit_entry.pack(side=LEFT, fill=X, expand=True, before=del_btn)
                edit_entry.focus_set()
                edit_entry.selection_range(0, tk.END)

                def save_edit(event):
                    new_value = entry_var.get().strip()
                    if new_value:
                        item["value"] = new_value
                        lbl.config(text=new_value)

                    edit_entry.destroy()
                    lbl.pack(side=LEFT, fill=X, expand=True, before=del_btn)

                edit_entry.bind("<Return>", save_edit)
                edit_entry.bind("<FocusOut>", save_edit)

            label.bind("<Double-1>", lambda e, item=arg_item, lbl=label, frame=row_frame, btn=delete_btn: start_edit(e, item, lbl, frame, btn))
            ToolTip(label, "Double-click to edit this argument.")
            label.pack(side=LEFT, fill=X, expand=True, anchor=W)

class FlextabsApp:
    """Main application controller for flextabs UI."""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LLaMA Server GUI")
        self.root.geometry("1080x720")
        self.root.minsize(1080, 720)

        self.server_manager = ServerManager()
        self.tray_manager = TrayManager(self)
        self.tab_states = {}

        ServerTabContent.server_manager = self.server_manager
        ServerTabContent.app = self

        tab_configs = [
            TabConfig(
                id="server",
                title="New Server",
                content_class=ServerTabContent,
                icon="🖥️",
                tooltip="Add a new Server management tab",
                closable=True
            ),
            TabConfig(
                id="settings",
                title="Settings",
                content_class=SettingsTabContent,
                icon="⚙️",
                tooltip="Global application settings",
                keyboard_shortcut="<Control-s>"
            ),
            TabConfig(
                id="help",
                title="Help",
                content_class=HomeTabContent,
                icon="ℹ️",
                tooltip="Help and documentation",
                closable=True
            )
        ]

        self.tab_manager = TabManager(
            parent=self.root,
            tab_configs=tab_configs,
            opener_type="toolbar",
            opener_config={
                "position": "top",
                "width": 150,
                "title": "Navigation"
            },
            close_button_style="both",
            close_mode="active_only",
            enable_keyboard_shortcuts=True,
            show_notebook_icons=True
        )
        self.tab_manager.pack(fill=tk.BOTH, expand=True)
        self.tab_manager.open_tab("help")

        self.root.protocol("WM_DELETE_WINDOW", self.on_window_close)

    def register_tab_state(self, tab_id, state):
        self.tab_states[tab_id] = state

    def get_app_dir(self):
        import os
        from core.utils import get_config_path
        return os.path.dirname(get_config_path(""))

    def ensure_configs_dir(self, app_dir):
        import os
        configs_dir = os.path.join(app_dir, "configs")
        os.makedirs(configs_dir, exist_ok=True)
        return configs_dir



    def on_window_close(self):
        if self.server_manager.any_server_running():
            choice = self._close_choice_dialog()
            if choice == "exit":
                self.quit_application()
            elif choice == "minimize":
                self.root.withdraw()
                self.tray_manager.show_tray()
            else:
                return
        else:
            self.quit_application()

    def _close_choice_dialog(self) -> str:
        dialog = ttk.Toplevel(self.root)
        dialog.title("Close LLaMA Server GUI")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(
            dialog,
            text="One or more servers are still running.",
            padding=(15, 10, 15, 0)
        ).pack(anchor=tk.W)

        ttk.Label(
            dialog,
            text="What would you like to do?",
            padding=(15, 0, 15, 10)
        ).pack(anchor=tk.W)

        choice = {"value": "cancel"}

        btn_frame = ttk.Frame(dialog, padding=(10, 0, 10, 10))
        btn_frame.pack(fill=tk.X)

        def set_choice(value: str):
            choice["value"] = value
            dialog.destroy()

        ttk.Button(btn_frame, text="Exit", command=lambda: set_choice("exit"), bootstyle="danger").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Minimize", command=lambda: set_choice("minimize"), bootstyle="secondary").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Cancel", command=lambda: set_choice("cancel"), bootstyle="light").pack(side=tk.RIGHT)

        # Center on parent window
        dialog.update_idletasks()
        self.root.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (dialog.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        dialog.wait_window()
        return choice["value"]

    def show_window(self):
        self.root.after(0, self.root.deiconify)

    def open_browser(self):
        # Open browser for the first running server, if any
        for state in self.tab_states.values():
            if state.is_running:
                config = ServerTabContent.app.tab_states[state.server_id]
                return True
        return False

    def quit_application(self):
        self.server_manager.terminate_all_servers()
        self.tray_manager.hide_tray()
        self.root.after(0, self.root.destroy)

    def any_server_running(self):
        return self.server_manager.any_server_running()

    def run(self):
        self.root.mainloop()