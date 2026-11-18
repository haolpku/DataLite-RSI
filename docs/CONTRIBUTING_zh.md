# DataFlow-RSI 中文贡献指南

DataFlow-RSI 接受四类可以彼此独立提交的贡献：Benchmark、Dataset、实验结果和
RSI 方法代码。完整规则以仓库根目录的
[`CONTRIBUTING.md`](../CONTRIBUTING.md) 为准。

## 贡献 Benchmark

在 `benchmarks/<benchmark-id>/` 下提交：

- `benchmark.json`：机器可读的任务、数据、评测器和指标定义；
- `README.md`：任务背景、指标解释、限制、污染和许可证说明；
- 评测器源码及不依赖完整数据的小型测试。

从 [`templates/benchmark.json`](../templates/benchmark.json) 复制模板。大文件
放 Hugging Face，GitHub 只保存引用和评测代码。

## 贡献 Dataset

先将数据发布到 Hugging Face，再在 `datasets/<dataset-id>/` 中登记：

- Hugging Face 仓库 ID；
- 固定的 commit revision，不要只写 `main`；
- 数据来源、生成或采集过程；
- splits、模态、格式和规模；
- license、使用限制、隐私、PII、偏差和安全说明。

从 [`templates/dataset.json`](../templates/dataset.json) 复制模板。不要把数据压缩包、
模型权重或媒体集合提交进 GitHub。

## 贡献实验结果

在 `results/submissions/<submission-id>/result.json` 中提交结构化结果，并将
`status` 设为 `unverified`。需要同时记录：

- Benchmark ID 和版本；
- 模型与 RSI 方法；
- baseline 与 RSI 后结果；
- 采样参数、随机种子、迭代次数和资源预算；
- 代码 commit、数据 commit、容器镜像 digest；
- 完整复现命令和大体积结果文件的外部地址。

请勿直接修改 `results/leaderboard.json`；维护者验证后统一生成榜单。

## 贡献 RSI 方法代码

在 `rsi/methods/<method-id>/` 中提交 `method.json`、方法说明、源码、默认配置
和测试。方法说明需要明确：

- 每轮递归修改什么对象；
- 使用什么反馈信号和接受规则；
- 停止条件及计算预算；
- 是否有人类干预；
- 安全约束、回滚机制和已知失败模式。

## Docker 约定

`Dockerfile` 和构建配置放 GitHub 的 `docker/`，构建后的镜像放 GHCR。模型
权重和 Benchmark 数据应在运行时下载或挂载，不能打进镜像；API Key 只能通过
运行时环境变量或 secret 注入。

## 提交前检查

```bash
python scripts/validate_contributions.py .
python -m unittest discover -s tests -v
```

目录名和 manifest 中的 `id` 必须一致，统一使用小写 `kebab-case`。一次 PR
尽量只解决一种主要贡献，方便科学审查和独立验证。
