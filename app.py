import fasttext
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# --- 配置 ---
# 映射模型名称到实际路径。注意：lid.176.bin 和 lid.176.ftz 假定在 'model/' 目录下
MODEL_PATHS = {
    "lid.176.bin": "model/lid.176.bin",
    "lid.176.ftz": "model/lid.176.ftz",
}
MODELS = {}  # 字典用于存储加载的模型实例

# 初始化 Flask 应用
app = Flask(__name__)
# 允许跨域请求 (CORS)
CORS(app)


def load_models():
    """在应用启动时加载所有配置的模型"""
    global MODELS
    print("--- 正在加载 FastText 模型 ---")

    # 确保 'model' 目录存在，如果模型路径包含它
    if not os.path.exists("model") and any("model/" in path for path in MODEL_PATHS.values()):
        print("警告: 'model/' 目录不存在。请确保模型文件已放置在正确的位置。")

    loaded_count = 0
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                # 显式加载，如果模型不存在或损坏，将抛出 ValueError
                MODELS[name] = fasttext.load_model(path)
                print(f"   ✅ 模型 '{name}' 加载成功。")
                loaded_count += 1
            except ValueError as e:
                print(f"   ❌ 模型 '{name}' 加载失败: {e}")
        else:
            print(f"   ❌ 文件 '{path}' 不存在，跳过加载。")

    if loaded_count == 0:
        print("致命错误: 未成功加载任何模型。请检查文件路径和名称。")
        global model
        model = None

    # 在应用第一次请求前加载模型


with app.app_context():
    load_models()


@app.route('/')
def index():
    """提供前端 HTML 文件"""
    # 假设 index.html 和 app.py 在同一目录
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    API 接口：接收文本和模型名称，返回语种预测结果
    请求体: {"text": "待检测的文本", "model_name": "lid.176.bin"}
    响应体: {"lang_code": "en", "confidence": 0.99}
    """
    if not MODELS:
        return jsonify({"error": "No model loaded. Check server logs."}), 500

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Invalid request: 'text' field missing."}), 400

    text = data['text']
    # 从请求中获取模型名称，默认使用用户自己的模型
    model_name = data.get('model_name', 'lid_model.bin')

    if model_name not in MODELS:
        return jsonify({"error": f"Model '{model_name}' not found or failed to load."}), 404

    # 根据模型名称获取已加载的模型实例
    model = MODELS[model_name]

    if not text.strip():
        return jsonify({"error": "Input text cannot be empty."}), 400

    try:
        # FastText 预测，k=1 只返回最可能的 Top 1 结果
        labels, probs = model.predict([text], k=1)

        # 提取结果并清理标签
        lang_code = labels[0][0].replace('__label__', '')
        confidence = probs[0][0]

        return jsonify({
            "lang_code": lang_code,
            "confidence": round(float(confidence), 4)  # 转换为标准浮点数并保留4位小数
        })
    except Exception as e:
        # 捕获预测过程中的异常
        print(f"预测过程中发生错误: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)