import os
from openai import OpenAI
from my_lib.dp_lib import config

# 从环境变量获取 DeepSeek API Key
api_key = config.Settings().deepseek_api_key
if not api_key:
    raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

# 初始化 OpenAI 客户端（假设 DeepSeek 的 API 兼容 OpenAI 格式）
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1",  # DeepSeek API 的基地址
)

# 定义提示词
prompt = """请帮我用 HTML 生成一个五子棋游戏，HTML 页面为古典主题背景,  页面中包含竹叶，HTML 中用英语说明； 最终所有代码都保存在一个 HTML 中。"""

try:
    # 调用 DeepSeek Chat API
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个专业的 Web 开发助手，擅长用 HTML/CSS/JavaScript 编写游戏。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        stream=False
    )

    # 提取生成的 HTML 内容
    if response.choices and len(response.choices) > 0:
        html_content = response.choices[0].message.content

        # 保存到文件
        with open("gomoku_v3.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        print("五子棋游戏已保存为 gomoku.html")
    else:
        print("未收到有效响应")

except Exception as e:
    print(f"调用 API 出错: {e}")
