# LLM Self-Dialogue PoC Runner

This project is a Proof of Concept (PoC) runner that enables Large Language Models (LLMs) to engage in self-dialogue. You provide an initial prompt, and the script makes the model respond to its own outputs for a specified number of cycles.

The runner supports both local GGUF models (via `llama-cpp-python`) and remote models through the Google Gemini API.

## Features

- **Self-Dialogue Loop**: Automatically feeds a model's output back to it as a new prompt.
- **Multi-Model Support**: Run experiments on different models simultaneously.
- **Local & API Models**: Supports local GGUF models for offline use and Gemini API for powerful cloud-based models.
- **Configurable**: Easily configure models, prompts, and experiment parameters via a YAML file.
- **JSONL I/O**: Uses JSON Lines format for easy reading and writing of inputs and results.

## How It Works

1.  **Configuration**: The script reads `config.yaml` to get the list of models to run, the path to the input prompts, and other settings.
2.  **Initialization**: It initializes the specified model runners (e.g., a local Llama 3 or a Gemini model).
3.  **Input Loading**: It loads the initial prompts from the `inputs.jsonl` file.
4.  **Dialogue Loop**: For each prompt and for each model, it starts a dialogue loop:
    a. The initial prompt is sent to the model.
    b. The model's response is recorded.
    c. The response is then used as the next prompt for the model.
    d. This continues for the number of `cycles` defined in the configuration.
5.  **Output**: The entire conversation history for each model is saved to a separate `.jsonl` file in the `results/` directory.

## Getting Started

### 1. Prerequisites

- Python 3.8+
- For local GGUF models: A C++ compiler and the necessary build tools for `llama-cpp-python`. Follow their installation guide for your specific OS (e.g., Metal for macOS, CUDA for NVIDIA).
- For Gemini models: A Google Cloud project with the Gemini API enabled and an API key.

### 2. Installation

Clone the repository and install the required Python packages:

```bash
git clone <repository_url>
cd <repository_name>
pip install -r requirements.txt
If you are using a Mac with Apple Silicon for a local GGUF model, you may need to install llama-cpp-python with Metal support:

# Uninstall existing version first if you installed from requirements.txt
pip uninstall llama-cpp-python -y

# Install with Metal support
CMAKE_ARGS="-DLLAMA_METAL=on" pip install -U llama-cpp-python --no-cache-dir
3. Configuration

a. config.yaml

This is the main configuration file. An example config.yaml is provided.

models: A list of models to test.

name: A unique name for the model (used for the output file).
type: Either gguf for local models or gemini for the API.
path: (GGUF only) The absolute path to your .gguf model file.
model: (Gemini only) The model name, e.g., gemini-1.5-flash.
You can also set generation parameters like temperature, top_p, and max_tokens.
data.inputs: The path to your input prompts file (e.g., inputs.jsonl).
output_dir: The directory where result files will be saved.
cycles: The number of times the model should respond to itself.
b. Set Environment Variables (for Gemini)

If you are using the Gemini API, set your API key as an environment variable:

export GEMINI_API_KEY="YOUR_API_KEY"
c. inputs.jsonl

This file contains the starting prompts for the dialogues. It must be a JSON Lines file, where each line is a valid JSON object. Each object should have at least a prompt key.

Example inputs.jsonl:

{"id": "dialogue_1", "prompt": "AIの意識について、あなた自身の考えを述べてください。"}
{"id": "dialogue_2", "prompt": "宇宙旅行が一般的になった未来の社会について、物語を創作してください。"}
4. Running the Script

Once your config.yaml is set up and your inputs.jsonl is ready, run the script:

python poc_runner.py
The script will display progress bars for each model as it processes the prompts.

5. Checking the Output

The results will be saved in the directory specified by output_dir (default: results/). A separate <model_name>.jsonl file will be created for each model defined in your configuration.

Each line in the output file represents one cycle of the dialogue and contains:

id: The ID from the input file.
cycle: The cycle number (1-indexed).
input: The initial prompt (only present in the first cycle).
output: The model's generated text for that cycle.
execution_time: Time taken for the generation.
and other metadata.

**`requirements.txt`**
PyYAML tqdm llama-cpp-python google-generativeai


**`config.yaml`**
```yaml
# ===============================================================
# PoC Runner Configuration
# ===============================================================

# ---------------------------------------------------------------
# 1. モデル設定 (models)
# ---------------------------------------------------------------
# 実行したいLLMをリスト形式で定義します。
# 複数のモデルを同時に実行できます。
models:
  # --- 例1: ローカルGGUFモデル ---
  # llama-cpp-pythonを使ってローカルでGGUF形式のモデルを動かす場合の設定
  - name: "Llama3-8B-Instruct-GGUF"  # 結果ファイル名などに使われるモデルの識別名
    type: "gguf"                      # モデル種別（'gguf' or 'gemini'）
    path: "/path/to/your/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf" # GGUFモデルファイルへの絶対パス
    # 以下はオプションのパラメータ
    n_ctx: 4096                       # コンテキストウィンドウのサイズ
    n_gpu_layers: -1                  # GPUにオフロードするレイヤー数 (-1で全て)
    temperature: 0.7                  # 生成のランダム性を制御 (0.0 - 2.0)
    top_p: 0.9                        # Top-pサンプリング
    max_tokens: 2048                  # 1回の応答で生成する最大トークン数

  # --- 例2: Gemini API ---
  # GoogleのGemini APIを利用する場合の設定
  - name: "Gemini-1.5-Flash"
    type: "gemini"
    model: "gemini-1.5-flash" # Geminiのモデル名 (e.g., 'gemini-1.5-pro', 'gemini-1.0-pro')
    # 以下はオプションのパラメータ
    temperature: 0.8
    top_p: 0.95
    max_tokens: 4096

# ---------------------------------------------------------------
# 2. データ設定 (data)
# ---------------------------------------------------------------
# 対話の開始点となるプロンプトを定義したファイルへのパス
data:
  inputs: "inputs.jsonl" # プロンプトがJSONL形式で記述されたファイル

# ---------------------------------------------------------------
# 3. 出力設定 (output)
# ---------------------------------------------------------------
# 結果を出力するディレクトリ
output_dir: "results"

# ---------------------------------------------------------------
# 4. 自己対話サイクル数 (cycles)
# ---------------------------------------------------------------
# 1つのプロンプトに対して自己対話を繰り返す回数
cycles: 5
inputs.jsonl

{"id": "dialogue_1", "prompt": "AIの意識について、あなた自身の考えを述べてください。"}
{"id": "dialogue_2", "prompt": "宇宙旅行が一般的になった未来の社会について、物語を創作してください。"}
Please review these files. Let me know if you would like any changes or if you are happy with them.
