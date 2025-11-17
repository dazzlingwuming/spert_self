# SpERT: Span-based Entity and Relation Transformer

## 项目简介
SpERT是一个基于跨度（Span）的实体和关系抽取模型，采用Transformer预训练技术，支持从文本中联合抽取实体和关系，适用于构建领域知识图谱等场景。本项目基于PyTorch实现，论文参见：[Span-based Joint Entity and Relation Extraction with Transformer Pre-training](https://arxiv.org/abs/1909.07755)（发表于ECAI 2020）。


## 项目结构
```
spert_self/
├── configs/               # 模型配置文件（如bert_base.json）
├── data/                  # 数据集存放路径
│   └── datasets/          # 各数据集（如CoNLL04、SciERC、ADE）
├── scripts/               # 辅助脚本
│   ├── fetch_datasets.sh  # 数据集下载脚本
│   ├── fetch_models.sh    # 预训练模型 checkpoint 下载脚本
│   └── conversion/        # 数据集格式转换脚本（如convert_ade.py）
├── spert/                 # 模型核心代码
│   ├── entities.py        # 实体相关类定义
│   ├── evaluator.py       # 评估器实现
│   ├── input_reader.py    # 输入数据读取
│   ├── spert.py           # 模型主逻辑（训练/评估/预测）
│   ├── templates/         # 评估结果可视化模板（HTML）
│   └── util.py            # 工具函数（如张量处理、路径操作）
├── py2neo_demo/           # Neo4j 知识图谱导入示例
│   └── neo4j_test.py      # Neo4j 连接与操作示例
├── args.py                # 命令行参数配置
├── config_reader.py       # 配置文件解析
├── LICENSE                # MIT 许可证
├── README.md              # 项目说明文档
└── requirements.txt       # 依赖库列表
```


## 环境搭建

### 依赖安装
项目依赖以下库，建议使用Python 3.5+环境：
```bash
pip install -r requirements.txt
```
主要依赖说明：
- PyTorch 1.4.0（深度学习框架）
- transformers 4.1.1（预训练模型工具）
- scikit-learn 0.24.0（评估指标计算）
- tqdm 4.55.1（进度条显示）
- spacy 3.0.1（可选，用于文本分词）


### 数据与模型下载
1. **数据集下载**：包含CoNLL04、SciERC、ADE等公开数据集（已转换为项目所需JSON格式）：
   ```bash
   bash ./scripts/fetch_datasets.sh
   ```

2. **预训练模型 checkpoint 下载**：包含各数据集上训练好的最佳模型（5次运行结果）：
   ```bash
   bash ./scripts/fetch_models.sh
   ```


## 数据集准备与格式
### 自定义数据集
若使用自定义数据，需按照以下格式准备JSON文件：
```json
{
  "tokens": ["token1", "token2", ..., "tokenN"],  # 分词后的文本
  "entities": [
    {
      "start": 0,          # 实体起始 token 索引
      "end": 2,            # 实体结束 token 索引（包含）
      "type": "EntityType" # 实体类型（如"疾病"、"药物"）
    },
    ...
  ],
  "relations": [
    {
      "head": 0,           # 头实体在 entities 中的索引
      "tail": 1,           # 尾实体在 entities 中的索引
      "type": "RelationType" # 关系类型（如"治疗"、"导致"）
    },
    ...
  ]
}
```

### 数据预处理
- 对于原始文本，可使用`spacy`进行分词（需下载对应模型，如`python -m spacy download en_core_web_sm`）。
- 中文文本可参考`spert/util.py`中的`creat_text2token`函数，按字符分割或使用中文分词工具。


## 模型配置
配置文件（如`configs/bert_base.json`）关键参数说明：
- `model_path`：预训练模型路径（如BERT、SciBERT）。
- `tokenizer_path`：分词器路径（需与模型匹配）。
- `neg_entity_count`：负例实体采样数量（可减少假阳性实体）。
- `max_span_size`：最大实体跨度（ token 数量）。
- `batch_size`：训练批次大小。
- `learning_rate`：学习率。


## 使用示例

### 1. 训练模型
以CoNLL04数据集为例，使用配置文件启动训练：
```bash
python ./spert.py train --config configs/example_train.conf
```

### 2. 评估模型
在测试集上评估已训练模型：
```bash
python ./spert.py eval --config configs/example_eval.conf
```

### 3. 预测新数据
对自定义文本进行实体和关系预测（需指定输入数据路径，参考`data/datasets/conll04/conll04_prediction_example.json`）：
```bash
python ./spert.py predict --config configs/example_predict.conf
```
- 若输入为原始文本，需确保`spacy`已安装并配置`spacy_model`参数。


## 知识图谱构建
结合实体和关系抽取结果，可将数据导入图数据库（如Neo4j）构建知识图谱：
1. 抽取结果导出为CSV或JSON格式。
2. 使用`py2neo_demo/neo4j_test.py`中的示例代码连接Neo4j：
   ```python
   from py2neo import GraphDatabase

   # 连接数据库
   driver = GraphDatabase.driver("bolt://localhost:7687", auth=("username", "password"))

   # 插入实体和关系
   with driver.session() as session:
       session.run("CREATE (d:疾病 {name: '感冒'})")
       session.run("CREATE (m:药物 {name: '阿司匹林'})")
       session.run("MATCH (d:疾病), (m:药物) WHERE d.name = '感冒' AND m.name = '阿司匹林' "
                   "CREATE (d)-[r:治疗]->(m)")
   ```


## 常见问题
- **假阳性实体过多**：增加配置文件中的`neg_entity_count`参数。
- **分词问题**：中文文本建议使用`spacy`中文模型或自定义分词逻辑（参考`spert/util.py`）。
- **版本兼容**：确保PyTorch与transformers版本匹配，建议使用requirements.txt中指定的版本。
- **训练中断**：模型会自动保存checkpoint，可通过`--resume`参数继续训练。


## 许可证
本项目基于MIT许可证开源，详情参见[LICENSE](LICENSE)。


## 参考文献
详见项目原README中的[References](README.md#References)部分。
