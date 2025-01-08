# llama.cpp Proxy

OpenAI API互換のllama.cppサーバー用リバースプロキシ。llama.cppサーバーをOpenAI APIと同じインターフェースで利用できるようにします。

## 機能

- OpenAI API互換エンドポイント (/v1/completions, /v1/chat/completions)
- 柔軟なチャットテンプレートのカスタマイズ
- API認証とレート制限
- ストリーミングレスポンス対応
- 文法制約機能 (llama.cppのgrammar機能)のサポート

## 必要条件

- Python 3.11以上
- 実行中のllama.cppサーバー

## インストール

```bash
pip install -e .
```

## 設定

環境変数:

```bash
# API認証用キー（少なくとも1つは必要）
UNLIMITED_API_KEY=your-unlimited-api-key  # レート制限なし
LIMITED_API_KEY=your-limited-api-key      # レート制限あり
```

## 使用方法

1. サーバーの起動:

```bash
llamacpp-proxy-server --llamacpp-server http://localhost:8080 --chat-template-jinja path/to/template.jinja
```

または、モジュールとして実行:

```bash
python -m llamacpp_proxy.main --llamacpp-server http://localhost:8080 --chat-template-jinja path/to/template.jinja
```

主なオプション:
- `--host`: バインドするホスト (デフォルト: 0.0.0.0)
- `--port`: バインドするポート (デフォルト: 8000)
- `--llamacpp-server`: llama.cppサーバーのURL (デフォルト: http://localhost:8080)
- `--chat-template-jinja`: チャットテンプレートファイルのパス
- `--rate-limit-window`: レート制限の時間窓（秒） (デフォルト: 60)
- `--rate-limit-max-requests`: 時間窓あたりの最大リクエスト数 (デフォルト: 10)

2. APIの利用:

```python
import openai

openai.api_key = "your-api-key"
openai.api_base = "http://localhost:8000/v1"

# チャット補完
response = openai.ChatCompletion.create(
    model="your-model",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

# テキスト補完
response = openai.Completion.create(
    model="your-model",
    prompt="Once upon a time",
    max_tokens=100
)
```

## テンプレートの設定

チャットテンプレートはJinja2形式で記述します。例：

```jinja
{%- if messages[0]['role'] == 'system' %}
 {%- set system_message = messages[0]['content'] %}
 {%- set loop_messages = messages[1:] %}
{%- else %}
 {%- set loop_messages = messages %}
{%- endif %}

{%- for message in loop_messages %}
 {%- if message['role'] == 'user' %}
 {{- '[INST] ' + message['content'] + ' [/INST]' }}
 {%- elif message['role'] == 'assistant' %}
 {{- ' ' + message['content'] + eos_token}}
 {%- endif %}
{%- endfor %}
```

## 開発

1. 依存関係のインストール:
```bash
pip install -e ".[test]"
```

2. テストの実行:
```bash
pytest
```

3. カバレッジレポートの生成:
```bash
pytest --cov --cov-report=html
```

## ライセンス

[Apache License 2.0](LICENSE)

このプロジェクトは[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)の下でライセンスされています。詳細については[LICENSE](LICENSE)ファイルを参照してください。