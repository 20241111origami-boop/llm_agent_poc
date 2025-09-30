import os
import json
import gc
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Union

import yaml
from tqdm import tqdm

# llama-cpp-pythonライブラリをインポート
from llama_cpp import Llama

# Google Generative AI (for Gemini)
import google.generativeai as genai

# ==============================================================================
# 1. 共通インターフェース
# ==============================================================================

class ModelRunner(ABC):
    def __init__(self, model_config: dict):
        self.config = model_config
        self.name = model_config['name']

    @abstractmethod
    def generate(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        """プロンプトから応答を生成。promptはstrまたは[{"role": "user", "content": "..."}]形式"""
        pass



    def cleanup(self):
        """リソースのクリーンアップ（必要に応じてオーバーライド）"""
        pass

# ==============================================================================
# 2. 各モデルの実装クラス
# ==============================================================================

class GGUFRunner(ModelRunner):
    """
    ローカルでllama-cpp-python経由でGGUFモデルを実行するランナー。
    MacのMetal GPUオフロードを前提とする。
    """
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        print(f"Initializing GGUF model: {self.name}...")
        
        model_path = model_config['path']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GGUF model file not found at: {model_path}")

        # 設定値を取得（デフォルト値付き）
        n_gpu_layers = model_config.get('n_gpu_layers', -1)
        n_ctx = model_config.get('n_ctx', 4096)
        n_threads = model_config.get('n_threads', None)  # Noneの場合は自動
        
        # Metal GPUに可能な限り多くのレイヤーをオフロードする設定
        llama_kwargs = {
            'model_path': model_path,
            'n_gpu_layers': n_gpu_layers,
            'n_ctx': n_ctx,
            'verbose': False
        }
        
        # スレッド数が指定されている場合のみ追加
        if n_threads is not None:
            llama_kwargs['n_threads'] = n_threads
            
        self.model = Llama(**llama_kwargs)
        print(f"GGUF model '{self.name}' loaded successfully.")

    def generate(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        try:
            # 生成パラメータを設定から取得
            max_tokens = self.config.get('max_tokens', 2048)
            temperature = self.config.get('temperature', 0.7)
            top_p = self.config.get('top_p', 0.9)
            
            # promptがstrの場合、messages形式に変換
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt
            
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error during generation with {self.name}: {e}")
            return f"ERROR: {str(e)}"

    def cleanup(self):
        """モデルのクリーンアップ"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        gc.collect()

class GeminiRunner(ModelRunner):
    """
    Google Generative AI SDK経由でGemini APIを呼び出すランナー。
    (自己対話ループが機能するように修正済み)
    """
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        print(f"Initializing Gemini model: {self.name}...")

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        genai.configure(api_key=api_key)

        # 生成設定をconfigから取得
        generation_config = {}
        if 'temperature' in model_config:
            generation_config['temperature'] = model_config['temperature']
        if 'top_p' in model_config:
            generation_config['top_p'] = model_config['top_p']
        if 'max_tokens' in model_config:
            generation_config['max_output_tokens'] = model_config['max_tokens']

        model_kwargs = {'model_name': model_config['model']}
        if generation_config:
            model_kwargs['generation_config'] = generation_config

        self.model = genai.GenerativeModel(**model_kwargs)
        print(f"Gemini model '{self.name}' initialized successfully.")
    def generate(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        """
        プロンプトから応答を生成。
        response.textアクセサが引き起こすエラーを回避するため、
        より安全に応答構造を解析するように修正済み。
        """
        try:
            # GGUFRunnerと処理を統一するため、promptがstrの場合はmessages形式に変換
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt

            # Gemini APIが要求する履歴フォーマットに変換
            # 'assistant' roleを 'model' roleにマッピングする
            gemini_history = []
            for msg in messages:
                role = 'model' if msg['role'] == 'assistant' else 'user'
                gemini_history.append({
                    'role': role,
                    'parts': [msg['content']]
                })

            # 履歴全体を渡してコンテンツを生成
            response = self.model.generate_content(gemini_history)

            # --- ここからが修正箇所 ---
            # 応答の構造をより安全に解析する
            # response.text は内部で candidates[0] を参照しようとして失敗するため、
            # 先に candidates の存在と内容を直接確認する。

            # 正常に応答テキストが存在するかを、短絡評価を利用して安全にチェック
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            else:
                # 応答テキストが存在しない場合、その理由を調査してエラーメッセージを返す
                finish_reason_name = "UNKNOWN"
                if response.candidates:
                    try:
                        # finish_reasonはenumなので、その名前を取得
                        finish_reason_name = response.candidates[0].finish_reason.name
                    except (AttributeError, IndexError):
                        pass  # 取得できなくても続行

                prompt_feedback_reason = "NONE"
                if hasattr(response, 'prompt_feedback') and str(response.prompt_feedback.block_reason) != "BLOCK_REASON_UNSPECIFIED":
                    try:
                        prompt_feedback_reason = response.prompt_feedback.block_reason.name
                    except AttributeError:
                        pass
                
                error_message = (
                    f"ERROR: Response generation failed or returned empty content. "
                    f"Finish Reason: {finish_reason_name}. "
                    f"Prompt Block Reason: {prompt_feedback_reason}."
                )
                # コンソールにも詳細を出力しておくとデバッグに役立つ
                print(f"Warning from {self.name}: {error_message}")
                return error_message
            # --- ここまでが修正箇所 ---

        except Exception as e:
            # generate_content自体が例外を発生させた場合
            print(f"Error during API call for {self.name}: {e}")
            return f"ERROR: {str(e)}"
    def cleanup(self):
        """
        GeminiRunnerでは特別なクリーンアップは不要だが、
        将来的な拡張（ChatSessionなど）のためにプレースホルダーとして定義。
        """
        pass


# ==============================================================================
# 3. 実行管理クラス
# ==============================================================================
class ExperimentManager:
    def __init__(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        self._validate_config()
        self.runners: List[ModelRunner] = []

    def _validate_config(self):
        """設定ファイルの基本的な検証"""
        if 'models' not in self.config:
            raise ValueError("Config must contain 'models' section")
        if 'data' not in self.config or 'inputs' not in self.config['data']:
            raise ValueError("Config must contain 'data.inputs' path")
        
        # 各モデル設定の検証
        for model_conf in self.config['models']:
            if 'name' not in model_conf:
                raise ValueError("Each model config must contain 'name' field")
            if 'type' not in model_conf:
                raise ValueError(f"Model config for '{model_conf['name']}' missing 'type' field")
            model_type = model_conf['type']
            if model_type == 'gguf' and 'path' not in model_conf:
                raise ValueError(f"GGUF model '{model_conf['name']}' missing 'path' field")
            elif model_type == 'gemini' and 'model' not in model_conf:
                raise ValueError(f"Gemini model '{model_conf['name']}' missing 'model' field")

    def _initialize_runners(self):
        """モデルランナーの初期化"""
        for model_conf in self.config['models']:
            try:
                model_type = model_conf.get('type')
                if not model_type:
                    print(f"Warning: Model config missing 'type' field. Skipping: {model_conf}")
                    continue
                    
                if model_type == 'gguf':
                    self.runners.append(GGUFRunner(model_conf))
                elif model_type == 'gemini':
                    self.runners.append(GeminiRunner(model_conf))
                else:
                    print(f"Warning: Unknown model type '{model_type}'. Skipping.")
            except Exception as e:
                print(f"Error initializing model {model_conf.get('name', 'unknown')}: {e}")
                continue

    def _load_inputs(self) -> List[Dict[str, Any]]:
        """入力データの読み込み"""
        input_path = Path(self.config['data']['inputs'])
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        inputs = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    inputs.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue
        
        if not inputs:
            raise ValueError("No valid inputs found in input file")
            
        return inputs


    def run(self):
        """実験の実行"""
        try:
            self._initialize_runners()

            if not self.runners:
                print("No valid model runners initialized. Exiting.")
                return

            inputs = self._load_inputs()
            results_dir = Path(self.config.get('output_dir', 'results'))
            results_dir.mkdir(exist_ok=True)

            for runner in self.runners:
                output_path = results_dir / f"{runner.name}.jsonl"
                print(f"\nRunning experiment for model: {runner.name}")
                print(f"Output will be saved to: {output_path}")

                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        for item in tqdm(inputs, desc=f"Processing prompts for {runner.name}"):
                            cycles = self.config.get("cycles", 10)
                            prompt = item.get("prompt")
                            history = [{"role": "user", "content": prompt}]

                            for c in range(cycles):
                                start_time = time.time()
                                output_text = runner.generate(history)
                                end_time = time.time()
                                execution_time = end_time - start_time
                                
                                # output_token_countの計算は、Grokの指摘通り不正確ですが、
                                # 今回の修正範囲外として一旦そのままにします。
                                output_token_count = len(output_text.split())

                                # まず結果を記録する
                                result_entry = {
                                    "id": item.get("id", None),
                                    "cycle": c + 1,
                                    "input": prompt if c == 0 else None,
                                    "output": output_text,
                                    "execution_time": execution_time,
                                    "output_token_count": output_token_count,
                                    "parameters": {
                                        "temperature": runner.config.get('temperature', 0.0),
                                        "top_p": runner.config.get('top_p', 0.9),
                                        "max_tokens": runner.config.get('max_tokens', 2048)
                                    }
                                }
                                f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

                                # --- ここからがGrokの提案に基づく修正箇所 ---

                                # 1. まず、モデルの応答を「assistant」として履歴に追加
                                history.append({"role": "assistant", "content": output_text})

                                # 2. 次に、エラーでないことを確認し、同じ応答を新しい「user」の
                                #    発言として追加することで、自己対話を継続させる
                                if output_text and not output_text.strip().startswith("ERROR:"):
                                    history.append({"role": "user", "content": output_text})
                                else:
                                    # エラーが発生した場合や応答が空の場合は、
                                    # この対話の連鎖を中断する
                                    print(f"  -> Cycle {c+1} resulted in an error or empty response. Stopping dialogue.")
                                    break
                                # --- ここまでが修正箇所 ---

                except Exception as e:
                    print(f"Error during experiment for {runner.name}: {e}")
                    continue

        finally:
            # リソースのクリーンアップ
            for runner in self.runners:
                try:
                    runner.cleanup()
                except Exception as e:
                    print(f"Error cleaning up {runner.name}: {e}")

# ==============================================================================
# 4. メイン実行ブロック
# ==============================================================================
if __name__ == "__main__":
    print("Starting PoC Runner...")
    try:
        manager = ExperimentManager(config_path="config.yaml")
        manager.run()
        print("\nAll experiments completed.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        exit(1)
