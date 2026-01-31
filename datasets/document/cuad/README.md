# CUAD 数据集

## 概述

**CUAD**（Contract Understanding Atticus Dataset）是由 The Atticus Project 发布的法律合同理解数据集，用于训练 AI 识别合同中的关键条款。

- **发布机构**: The Atticus Project
- **论文**: "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review" (NeurIPS 2021)
- **HuggingFace**: `theatticusproject/cuad-qa`
- **许可证**: CC BY 4.0
- **语言**: 英语

## 数据集特点

- **专家标注**: 由法律专业人士标注，保证专业准确性
- **41 类条款**: 涵盖企业交易中最重要的合同条款类型
- **真实合同**: 来自 EDGAR（SEC 电子数据收集系统）的真实商业合同
- **QA 格式**: 采用抽取式问答格式，便于训练

## 数据集规模

- **合同数量**: 510 份商业法律合同
- **标注数量**: 13,000+ 个专家标注
- **条款类型**: 41 类

## 41 类合同条款

| 类别 | 英文名 | 说明 |
|------|--------|------|
| 1 | Document Name | 文档名称 |
| 2 | Parties | 合同当事方 |
| 3 | Agreement Date | 协议日期 |
| 4 | Effective Date | 生效日期 |
| 5 | Expiration Date | 到期日期 |
| 6 | Renewal Term | 续约条款 |
| 7 | Notice Period To Terminate | 终止通知期 |
| 8 | Governing Law | 适用法律 |
| 9 | Most Favored Nation | 最惠国条款 |
| 10 | Non-Compete | 竞业禁止 |
| 11 | Exclusivity | 排他性条款 |
| 12 | No-Solicit Of Customers | 禁止招揽客户 |
| 13 | No-Solicit Of Employees | 禁止招揽员工 |
| 14 | Non-Disparagement | 禁止贬损 |
| 15 | Termination For Convenience | 便利终止 |
| 16 | Rofr/Rofo/Rofn | 优先权条款 |
| 17 | Change Of Control | 控制权变更 |
| 18 | Anti-Assignment | 禁止转让 |
| 19 | Revenue/Profit Sharing | 收益分成 |
| 20 | Price Restrictions | 价格限制 |
| 21 | Minimum Commitment | 最低承诺 |
| 22 | Volume Restriction | 数量限制 |
| 23 | IP Ownership Assignment | 知识产权转让 |
| 24 | Joint IP Ownership | 知识产权共有 |
| 25 | License Grant | 许可授予 |
| 26 | Non-Transferable License | 不可转让许可 |
| 27 | Affiliate License | 关联方许可 |
| 28 | Unlimited/All-You-Can-Eat License | 无限许可 |
| 29 | Irrevocable License | 不可撤销许可 |
| 30 | Source Code Escrow | 源代码托管 |
| 31 | Post-Termination Services | 终止后服务 |
| 32 | Audit Rights | 审计权 |
| 33 | Uncapped Liability | 无上限责任 |
| 34 | Cap On Liability | 责任上限 |
| 35 | Liquidated Damages | 违约金 |
| 36 | Warranty Duration | 保修期限 |
| 37 | Insurance | 保险条款 |
| 38 | Covenant Not To Sue | 不起诉承诺 |
| 39 | Third Party Beneficiary | 第三方受益人 |
| 40 | Confidentiality Duration | 保密期限 |
| 41 | Competitive Restriction | 竞争限制 |

## 数据字段说明

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `id` | string | 问题唯一标识符 |
| `title` | string | 合同文件名 |
| `context` | string | 合同文本段落 |
| `question` | string | 关于特定条款的问题 |
| `answers` | dict | 答案信息（与 SQuAD 格式相同） |

### answers 字段结构

```json
{
  "text": ["条款文本内容"],
  "answer_start": [起始位置]
}
```

## 数据示例

```json
{
  "id": "CreditAgreement_0001193125-18-227764_1",
  "title": "CreditAgreement_0001193125-18-227764",
  "context": "This CREDIT AGREEMENT dated as of July 25, 2018...",
  "question": "Highlight the parts (if any) of this contract related to Agreement Date.",
  "answers": {
    "text": ["July 25, 2018"],
    "answer_start": [35]
  }
}
```

## 评测指标

- **AUPR (Area Under Precision-Recall Curve)**: 主要评测指标
- **Precision/Recall**: 精确率和召回率
- 由于法律合同审查对漏检（false negative）更敏感，召回率尤为重要

## 使用方法

```python
from datasets import load_dataset

# 加载 QA 格式数据集
dataset = load_dataset("theatticusproject/cuad-qa")

# 查看示例
print(dataset["train"][0])

# 按条款类型筛选
governing_law = dataset["train"].filter(
    lambda x: "Governing Law" in x["question"]
)
```

## 应用场景

1. **合同审查自动化**: 自动识别合同中的关键条款
2. **法律 AI 助手**: 辅助律师进行合同审阅
3. **合规检查**: 检查合同是否包含必要条款
4. **尽职调查**: 加速 M&A 交易中的合同审查

## 预训练模型

官方提供了在 CUAD 上微调的模型：
- RoBERTa-base (~100M 参数)
- RoBERTa-large (~300M 参数)
- DeBERTa-xlarge (~900M 参数)

## 参考链接

- 官网: https://www.atticusprojectai.org/cuad
- GitHub: https://github.com/TheAtticusProject/cuad
- 论文: https://arxiv.org/abs/2103.06268
- HuggingFace: https://huggingface.co/datasets/theatticusproject/cuad-qa
