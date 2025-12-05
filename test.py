import fasttext
import sys
import os

# --- 配置 ---
# MODEL_PATH = 'model/lid_model.bin'  # 确保模型文件路径正确
MODEL_PATH = 'model/lid.176.bin'  # 确保模型文件路径正确
# MODEL_PATH = 'model/lid.176.ftz'  # 确保模型文件路径正确
TOP_K = 1  # 只显示最可能的 Top 1 预测结果


def load_and_test_model():
    """加载模型并进入交互式测试循环"""

    # 1. 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 '{MODEL_PATH}'。")
        print("请确保模型已训练完成，并且文件位于当前目录下。")
        sys.exit(1)

    try:
        # 2. 加载 FastText 模型
        model = fasttext.load_model(MODEL_PATH)
        print("--- ✅ FastText 模型加载成功 ---")
        print(f"   模型路径: {MODEL_PATH}")
        # 尝试获取标签数量，确认模型是分类模型
        num_labels = len(model.get_labels())
        print(f"   可识别语言数量: {num_labels}")
        print("-----------------------------------")

    except ValueError as e:
        print(f"加载模型时发生错误：{e}")
        print("请确认模型文件没有损坏。")
        sys.exit(1)

    # 3. 交互式测试循环
    print("开始实时语种检测 (输入 'exit' 或 'quit' 退出):")

    while True:
        try:
            # 获取用户输入
            user_input = input(">> 输入文本: ").strip()

            if user_input.lower() in ['exit', 'quit']:
                print("\n程序退出。")
                break

            if not user_input:
                continue

            # 进行预测
            # model.predict 接受一个文本列表
            predictions = model.predict([user_input], k=TOP_K)

            labels = predictions[0][0]
            probs = predictions[1][0]

            # 格式化输出结果
            results = []
            for label, prob in zip(labels, probs):
                lang_code = label.replace('__label__', '')
                results.append(f"**{lang_code}** (置信度: {prob:.4f})")

            print(f"   -> 预测结果 ({TOP_K}): {', '.join(results)}")

        except KeyboardInterrupt:
            # 捕获 Ctrl+C 信号
            print("\n程序退出。")
            break
        except Exception as e:
            print(f"发生未知错误: {e}")
            break


if __name__ == "__main__":
    load_and_test_model()