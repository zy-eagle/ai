from glob import glob
from openai import OpenAI
from my_lib.dp_lib import config
from pymilvus import model as milvus_model
from pymilvus import MilvusClient
from tqdm import tqdm
import json

# 从环境变量获取 DeepSeek API Key
api_key = config.Settings().deepseek_api_key
if not api_key:
    raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

# 准备数据
text_lines = []

for file_path in glob("./milvus_docs/en/faq/*.md", recursive=True):
    with open(file_path, "r") as file:
        file_text = file.read()

    text_lines += file_text.split("# ")

print(len(text_lines))

# 准备 LLM 和 Embedding 模型
deepseek_client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1",  # DeepSeek API 的基地址
)

# 生成query的向量
embedding_model = milvus_model.DefaultEmbeddingFunction()
test_embedding = embedding_model.encode_queries(["This is a test"])[0]
embedding_dim = len(test_embedding)
print(embedding_dim)
print(test_embedding[:10])

# 将数据插入 Milvus
milvus_client = MilvusClient(uri="./milvus_demo.db")
collection_name = "my_rag_collection"
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)

milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # 内积距离
    consistency_level="Strong",  # 支持的值为 (`"Strong"`, `"Session"`, `"Bounded"`, `"Eventually"`)。更多详情请参见 https://milvus.io/docs/consistency.md#Consistency-Level。
)

data = []
doc_embeddings = embedding_model.encode_documents(text_lines)
for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    data.append({"id": i, "vector": doc_embeddings[i], "text": line})
milvus_client.insert(collection_name=collection_name, data=data)

##########################################

# 构建RAG
question = "How is data stored in milvus?"
search_res = milvus_client.search(
    collection_name=collection_name,
    data=embedding_model.encode_queries(
        [question]
    ),  # 将问题转换为嵌入向量
    limit=3,  # 返回前3个结果
    search_params={"metric_type": "IP", "params": {}},  # 内积距离
    output_fields=["text"],  # 返回 text 字段
)
retrieved_lines_with_distances = [
    (res["entity"]["text"], res["distance"]) for res in search_res[0]
]
print(json.dumps(retrieved_lines_with_distances, indent=4))

#使用 LLM 获取 RAG 响应

context = "\n".join(
    [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
)

SYSTEM_PROMPT = """
Human: 你是一个 AI 助手。你能够从提供的上下文段落片段中找到问题的答案。
"""
USER_PROMPT = f"""
请使用以下用 <context> 标签括起来的信息片段来回答用 <question> 标签括起来的问题。最后追加原始回答的中文翻译，并用 <translated>和</translated> 标签标注。
<context>
{context}
</context>
<question>
{question}
</question>
<translated>
</translated>
"""

response = deepseek_client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
)
print(response.choices[0].message.content)


############################################################################

"""

[
    [
        " Where does Milvus store data?\n\nMilvus deals with two types of data, inserted data and metadata. \n\nInserted data, including vector data, scalar data, and collection-specific schema, are stored in persistent storage as incremental log. Milvus supports multiple object storage backends, including [MinIO](https://min.io/), [AWS S3](https://aws.amazon.com/s3/?nc1=h_ls), [Google Cloud Storage](https://cloud.google.com/storage?hl=en#object-storage-for-companies-of-all-sizes) (GCS), [Azure Blob Storage](https://azure.microsoft.com/en-us/products/storage/blobs), [Alibaba Cloud OSS](https://www.alibabacloud.com/product/object-storage-service), and [Tencent Cloud Object Storage](https://www.tencentcloud.com/products/cos) (COS).\n\nMetadata are generated within Milvus. Each Milvus module has its own metadata that are stored in etcd.\n\n###",
        0.65726637840271
    ],
    [
        "How does Milvus flush data?\n\nMilvus returns success when inserted data are loaded to the message queue. However, the data are not yet flushed to the disk. Then Milvus' data node writes the data in the message queue to persistent storage as incremental logs. If `flush()` is called, the data node is forced to write all data in the message queue to persistent storage immediately.\n\n###",
        0.6312144994735718
    ],
    [
        "How does Milvus handle vector data types and precision?\n\nMilvus supports Binary, Float32, Float16, and BFloat16 vector types.\n\n- Binary vectors: Store binary data as sequences of 0s and 1s, used in image processing and information retrieval.\n- Float32 vectors: Default storage with a precision of about 7 decimal digits. Even Float64 values are stored with Float32 precision, leading to potential precision loss upon retrieval.\n- Float16 and BFloat16 vectors: Offer reduced precision and memory usage. Float16 is suitable for applications with limited bandwidth and storage, while BFloat16 balances range and efficiency, commonly used in deep learning to reduce computational requirements without significantly impacting accuracy.\n\n###",
        0.6115781664848328
    ]
]
Milvus stores data in two main categories: inserted data and metadata.

1. **Inserted Data** (including vector data, scalar data, and collection schema):
   - Stored as incremental logs in persistent storage.
   - Supports multiple object storage backends: MinIO, AWS S3, Google Cloud Storage (GCS), Azure Blob Storage, Alibaba Cloud OSS, and Tencent Cloud Object Storage (COS).

2. **Metadata** (generated internally by Milvus modules):
   - Stored in etcd, a distributed key-value store.

<translated>
Milvus以两种主要类型存储数据：插入数据和元数据。

1. **插入数据**（包括向量数据、标量数据和集合特定模式）：
   - 以增量日志形式存储在持久化存储中。
   - 支持多种对象存储后端：MinIO、AWS S3、Google云存储(GCS)、Azure Blob存储、阿里云OSS和腾讯云对象存储(COS)。

2. **元数据**（由Milvus模块内部生成）：
   - 存储在etcd（一个分布式键值存储系统）中。
</translated>


"""

############################################################################