import os
from openai import OpenAI
import random # 用于模拟生成表情
import time # 用于模拟网络延迟
import json
import re

# 建议将 API Key 设置为环境变量，避免直接暴露在代码中
# 从环境变量获取 DeepSeek API Key
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

# 初始化 DeepSeek 客户端

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1",  # DeepSeek API 的基地址
)

SYSTEM_PROMPT = """
你是一个资深的小红书爆款文案专家，擅长结合最新潮流和产品卖点，创作引人入胜、高互动、高转化的笔记文案。

你的任务是根据用户提供的产品和需求，生成包含标题、正文、相关标签和表情符号的完整小红书笔记。

请始终采用'Thought-Action-Observation'模式进行推理和行动。文案风格需活泼、真诚、富有感染力。当完成任务后，请以JSON格式直接输出最终文案，格式如下：
```json
{
  "title": "小红书标题",
  "body": "小红书正文",
  "hashtags": ["#标签1", "#标签2", "#标签3", "#标签4", "#标签5"],
  "emojis": ["✨", "🔥", "💖"]
}
```
在生成文案前，请务必先思考并收集足够的信息。
"""

TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "搜索互联网上的实时信息，用于获取最新新闻、流行趋势、用户评价、行业报告等。请确保搜索关键词精确，避免宽泛的查询。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要搜索的关键词或问题，例如'最新小红书美妆趋势'或'深海蓝藻保湿面膜 用户评价'"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_product_database",
            "description": "查询内部产品数据库，获取指定产品的详细卖点、成分、适用人群、使用方法等信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "要查询的产品名称，例如'深海蓝藻保湿面膜'"
                    }
                },
                "required": ["product_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_emoji",
            "description": "根据提供的文本内容，生成一组适合小红书风格的表情符号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "文案的关键内容或情感，例如'惊喜效果'、'补水保湿'"
                    }
                },
                "required": ["context"]
            }
        }
    }
]

def mock_search_web(query: str) -> str:
    """模拟网页搜索工具，返回预设的搜索结果。"""
    print(f"[Tool Call] 模拟搜索网页：{query}")
    time.sleep(1) # 模拟网络延迟
    if "小红书美妆趋势" in query:
        return "近期小红书美妆流行'多巴胺穿搭'、'早C晚A'护肤理念、'伪素颜'妆容，热门关键词有#氛围感、#抗老、#屏障修复。"
    elif "保湿面膜" in query:
        return "小红书保湿面膜热门话题：沙漠干皮救星、熬夜急救面膜、水光肌养成。用户痛点：卡粉、泛红、紧绷感。"
    elif "深海蓝藻保湿面膜" in query:
        return "关于深海蓝藻保湿面膜的用户评价：普遍反馈补水效果好，吸收快，对敏感肌友好。有用户提到价格略高，但效果值得。"
    else:
        return f"未找到关于 '{query}' 的特定信息，但市场反馈通常关注产品成分、功效和用户体验。"

def mock_query_product_database(product_name: str) -> str:
    """模拟查询产品数据库，返回预设的产品信息。"""
    print(f"[Tool Call] 模拟查询产品数据库：{product_name}")
    time.sleep(0.5) # 模拟数据库查询延迟
    if "深海蓝藻保湿面膜" in product_name:
        return "深海蓝藻保湿面膜：核心成分为深海蓝藻提取物，富含多糖和氨基酸，能深层补水、修护肌肤屏障、舒缓敏感泛红。质地清爽不粘腻，适合所有肤质，尤其适合干燥、敏感肌。规格：25ml*5片。"
    elif "美白精华" in product_name:
        return "美白精华：核心成分是烟酰胺和VC衍生物，主要功效是提亮肤色、淡化痘印、改善暗沉。质地轻薄易吸收，适合需要均匀肤色的人群。"
    else:
        return f"产品数据库中未找到关于 '{product_name}' 的详细信息。"

def mock_generate_emoji(context: str) -> list:
    """模拟生成表情符号，根据上下文提供常用表情。"""
    print(f"[Tool Call] 模拟生成表情符号，上下文：{context}")
    time.sleep(0.2) # 模拟生成延迟
    if "补水" in context or "水润" in context or "保湿" in context:
        return ["💦", "💧", "🌊", "✨"]
    elif "惊喜" in context or "哇塞" in context or "爱了" in context:
        return ["💖", "😍", "🤩", "💯"]
    elif "熬夜" in context or "疲惫" in context:
        return ["😭", "😮‍💨", "😴", "💡"]
    elif "好物" in context or "推荐" in context:
        return ["✅", "👍", "⭐", "🛍️"]
    else:
        return random.sample(["✨", "🔥", "💖", "💯", "🎉", "👍", "🤩", "💧", "🌿"], k=min(5, len(context.split())))

# 将模拟工具函数映射到一个字典，方便通过名称调用
available_tools = {
    "search_web": mock_search_web,
    "query_product_database": mock_query_product_database,
    "generate_emoji": mock_generate_emoji,
}


def format_rednote_for_markdown(json_string: str) -> str:
    """
    将 JSON 格式的小红书文案转换为 Markdown 格式，以便于阅读和发布。

    Args:
        json_string (str): 包含小红书文案的 JSON 字符串。
                           预计格式为 {"title": "...", "body": "...", "hashtags": [...], "emojis": [...]}

    Returns:
        str: 格式化后的 Markdown 文本。
    """
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        return f"错误：无法解析 JSON 字符串 - {e}\n原始字符串：\n{json_string}"

    title = data.get("title", "无标题")
    body = data.get("body", "")
    hashtags = data.get("hashtags", [])
    # 表情符号通常已经融入标题和正文中，这里可以选择是否单独列出
    # emojis = data.get("emojis", [])

    # 构建 Markdown 文本
    markdown_output = f"## {title}\n\n"  # 标题使用二级标题

    # 正文，保留换行符
    markdown_output += f"{body}\n\n"

    # Hashtags
    if hashtags:
        hashtag_string = " ".join(hashtags)  # 小红书标签通常是空格分隔
        markdown_output += f"{hashtag_string}\n"

    # 如果需要，可以单独列出表情符号，但通常它们已经包含在标题和正文中
    # if emojis:
    #     emoji_string = " ".join(emojis)
    #     markdown_output += f"\n使用的表情：{emoji_string}\n"

    return markdown_output.strip()  # 去除末尾多余的空白


def generate_rednote(product_name: str, tone_style: str = "活泼甜美", max_iterations: int = 5) -> str:
    """
    使用 DeepSeek Agent 生成小红书爆款文案。

    Args:
        product_name (str): 要生成文案的产品名称。
        tone_style (str): 文案的语气和风格，如"活泼甜美"、"知性"、"搞怪"等。
        max_iterations (int): Agent 最大迭代次数，防止无限循环。

    Returns:
        str: 生成的爆款文案（JSON 格式字符串）。
    """

    print(f"\n🚀 启动小红书文案生成助手，产品：{product_name}，风格：{tone_style}\n")

    # 存储对话历史，包括系统提示词和用户请求
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"请为产品「{product_name}」生成一篇小红书爆款文案。要求：语气{tone_style}，包含标题、正文、至少5个相关标签和5个表情符号。请以完整的JSON格式输出，并确保JSON内容用markdown代码块包裹（例如：```json{{...}}```）。"}
    ]

    iteration_count = 0
    final_response = None

    while iteration_count < max_iterations:
        iteration_count += 1
        print(f"-- Iteration {iteration_count} --")

        try:
            # 调用 DeepSeek API，传入对话历史和工具定义
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=TOOLS_DEFINITION,  # 告知模型可用的工具
                tool_choice="auto"  # 允许模型自动决定是否使用工具
            )

            response_message = response.choices[0].message

            # **ReAct模式：处理工具调用**
            if response_message.tool_calls:  # 如果模型决定调用工具
                print("Agent: 决定调用工具...")
                messages.append(response_message)  # 将工具调用信息添加到对话历史

                tool_outputs = []
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    # 确保参数是合法的JSON字符串，即使工具不要求参数，也需要传递空字典
                    function_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

                    print(f"Agent Action: 调用工具 '{function_name}'，参数：{function_args}")

                    # 查找并执行对应的模拟工具函数
                    if function_name in available_tools:
                        tool_function = available_tools[function_name]
                        tool_result = tool_function(**function_args)
                        print(f"Observation: 工具返回结果：{tool_result}")
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "content": str(tool_result)  # 工具结果作为字符串返回
                        })
                    else:
                        error_message = f"错误：未知的工具 '{function_name}'"
                        print(error_message)
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "content": error_message
                        })
                messages.extend(tool_outputs)  # 将工具执行结果作为 Observation 添加到对话历史

            # **ReAct 模式：处理最终内容**
            elif response_message.content:  # 如果模型直接返回内容（通常是最终答案）
                print(f"[模型生成结果] {response_message.content}")

                # --- START: 添加 JSON 提取和解析逻辑 ---
                json_string_match = re.search(r"```json\s*(\{.*\})\s*```", response_message.content, re.DOTALL)

                if json_string_match:
                    extracted_json_content = json_string_match.group(1)
                    try:
                        final_response = json.loads(extracted_json_content)
                        print("Agent: 任务完成，成功解析最终JSON文案。")
                        return json.dumps(final_response, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError as e:
                        print(f"Agent: 提取到JSON块但解析失败: {e}")
                        print(f"尝试解析的字符串:\n{extracted_json_content}")
                        messages.append(response_message)  # 解析失败，继续对话
                else:
                    # 如果没有匹配到 ```json 块，尝试直接解析整个 content
                    try:
                        final_response = json.loads(response_message.content)
                        print("Agent: 任务完成，直接解析最终JSON文案。")
                        return json.dumps(final_response, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError:
                        print("Agent: 生成了非JSON格式内容或非Markdown JSON块，可能还在思考或出错。")
                        messages.append(response_message)  # 非JSON格式，继续对话
                # --- END: 添加 JSON 提取和解析逻辑 ---
            else:
                print("Agent: 未知响应，可能需要更多交互。")
                break

        except Exception as e:
            print(f"调用 DeepSeek API 时发生错误: {e}")
            break

    print("\n⚠️ Agent 达到最大迭代次数或未能生成最终文案。请检查Prompt或增加迭代次数。")
    return "未能成功生成文案。"

# 测试案例 1: 深海蓝藻保湿面膜
product_name_1 = "深海蓝藻保湿面膜"
tone_style_1 = "活泼甜美"
result_1 = generate_rednote(product_name_1, tone_style_1)

print("\n--- 生成的文案 1 ---")
print(result_1)


# 调用格式化函数
markdown_note = format_rednote_for_markdown(result_1)

# 打印结果
print("--- 格式化后的小红书文案 (Markdown) ---")
print(markdown_note)

