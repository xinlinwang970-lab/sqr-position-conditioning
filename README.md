# SQR 位置条件分离：独立实现与探索性实验

本仓库整理弦乐四声部分离的独立实现、配置、数据清单、历史逐条指标和复算工具。当前结果支持工程流程与探索性比较，研究结论受单随机种子、反复查看测试集、训练协议差异及数据来源不完全对应等限制。

参考论文是 Xu 等人的 [Position also matters! Separating Same Instruments in String Quartet using Timbral and Positional Cues](https://www.isca-archive.org/interspeech_2025/xu25b_interspeech.html)，Interspeech 2025。原始数据入口为 [String Quartet Set with Reverb](https://zenodo.org/records/14793971)。本仓库与原作者无隶属关系；文件名中的 `author_release` 表示本地实验使用作者发布的数据，不代表作者提供的模型或 checkpoint。

## 发布内容与解释范围

- `src/`、`scripts/`、`configs/`：模型与历史训练/评估入口，以及新增的可移植数据准备工具。
- `manifests/`：59 条 type2 训练、12 条验证、12 条测试记录；另保留历史 baseline 的 60/10/12 记录选择。
- `results/checkpoint_evals/`、`results/test_results/`：原始 11 份结果 JSON，未改动分数。
- `results/provenance/`：训练日志、历次可用测试评估、8 个 checkpoint 的本地审核校验值、原始配置和文件来源。
- `tests/`：模型、路由、数据读取、评估、合成训练及发布工具的测试。

仓库不打包音频、SoundFont、第三方数据归档或模型权重。已有 SQR 实验使用作者发布的空间化音频。论文描述的合成源来自 EnsembleSet/BBCSO；单独开展的 MuseScore 数据生成属于另一数据变体，不能用其名称替代这里的 SQR 数据来源。[论文数据说明](https://www.isca-archive.org/interspeech_2025/xu25b_interspeech.pdf)

**先阅读 [已知限制](docs/KNOWN_LIMITATIONS.md) 和 [数据来源](docs/DATA_PROVENANCE.md)。** 尤其是 E1cosine-B 的 checkpoint 指向自建 `quartet-data`，不可将其放入“相同 59 条 SQR 数据、只改变增强方式”的对照组。

## 安装与不依赖数据的检查

在下载后的仓库根目录，使用当前 Python 环境运行：

```bash
python -m pip install -e ".[torch,test]"
python -m pytest -q
python scripts/verify_results.py
```

若使用 GPU，请先按所需平台安装兼容的 PyTorch。`.[torch,test]` 会安装 PyTorch 与测试依赖；单独安装基础包不会安装 PyTorch。声明支持的 Python 范围不等于所有组合均已验证，实际本地验收环境记录于 `results/validation/`。GPU 完整训练和 Linux 环境未在本次整理中重新运行。

复算命令仅检查已有逐条分数、汇总算法和证据文件校验值，不会运行模型推理。

## 从 SQR 归档准备清单

自行从数据页面获取两份归档并解压到同一数据根目录。可以保留归档原有目录层级，无需把验证音频移到另一个文件夹。清单工具按仓库中冻结的记录 ID 选择音频，推导座位标签并核验 44.1 kHz、双声道和长度，计算完整 WAV 字节的 SHA-256。

```bash
python scripts/build_data_split_manifests.py --data-root data --output-dir manifests/local
```

验证曲目选择来自 [Zenodo 公布的六项清单](https://zenodo.org/records/14793971)。模板训练清单只包含 type2；验证和测试均含两种布局。工具不会移动、重采样或归一化音频。若实际音频与冻结清单不同，命令会停止；显式使用 `--allow-hash-differences` 时会生成差异报告，该数据必须另行标注为变体。

## 开训前检查与训练

```bash
python scripts/train_PureSwap_E1cosine-P15_1000ep.py --data-root data --train-manifest manifests/local/sqr_paper_split_train.jsonl --val-manifest manifests/local/sqr_paper_split_val.jsonl --test-manifest manifests/local/sqr_paper_split_test.jsonl --run-root runs/new_P15 --check-only
```

检查通过后，移除 `--check-only` 可开始该历史协议的训练。该协议固定 batch size=4、seed=2026、1000 轮目标与 p=0.15；完整训练成本较高。本次整理没有启动长时间训练。运行会保存 `release_launch.json`，记录实际入口、配置、清单路径及手工激活重计算设置。默认的自动恢复会接续输出目录内已有 checkpoint，因此新实验应使用新的 `--run-root`。

8 个训练入口均提供 `--help`、`--config`、`--data-root`、`--run-root`、三个清单路径和 `--check-only`。历史 baseline 另外要求显式 `--cache-root`。自建数据的 E1cosine-B 要求显式给出 train/val 清单和各数据根目录；本发布包无法恢复它原始的数据身份。

## 历史 baseline 的缓存入口

以下命令使用历史 60/5 项选择，生成保持输入样本幅度的缓存；这些选择清单从旧的归一化工作区找回，仅凭清单不能证明音频或缓存与旧训练完全一致。

```bash
python scripts/build_dense_cache.py --data-root data --output-root data/cache_baseline_new --train-manifest manifests/historical_baseline/train.jsonl --val-manifest manifests/historical_baseline/val.jsonl --test-manifest manifests/historical_baseline/test.jsonl
python scripts/train_author_release_12ep_warmup_cosine.py --data-root data --cache-root data/cache_baseline_new --run-root runs/new_baseline --check-only
```

缓存工具要求这些历史路径下的 WAV 已存在。若你的解压路径不同，请在副本清单里调整相对路径，再显式传入三个清单参数。工具不复制音频，也不执行论文的数据合成或 RMS 处理。

## 评估已有权重

需要自行提供与相应运行匹配的 checkpoint。原始权重校验值见 `results/provenance/checkpoint_verification.json`。

```bash
python scripts/evaluate_checkpoint.py --config configs/PureSwap_E1cosine-P15_1000ep.yaml --checkpoint weights/P15_checkpoint_025075.pt --manifest manifests/local/sqr_paper_split_test.jsonl --root data --output runs/new_eval/P15.json
```

评估采用历史整曲滑窗流程。指标是幅度敏感的波形 SDR：`10*log10(sum(reference**2)/(sum((estimate-reference)**2)+1e-8))`。它不执行 SI-SDR 的最优尺度投影，也不等于论文可能采用的其他 SDR 工具实现。历史 Hann 拼接端点和短输入行为均已记录，不能宣称与原论文分数可直接对标。

## 历史结果

单位 dB。Mean 为 48 条相关查询的均值；type1/type2 为各 24 条查询的中位数。`ep274` 是零起始索引，对应完成 275 轮。不同运行的数据、路由、调度器和预算不完全相同。

| 运行 | 已完成更新 | Mean | type1 中位数 | type2 中位数 |
|---|---:|---:|---:|---:|
| 历史 dense baseline，12 轮 | 52,236 | 1.3268 | -0.3299 | 1.6361 |
| E1cosine-B，自建数据来源待闭合，275 轮 | 16,225 | 1.6187 | 1.3918 | 1.4778 |
| E2，275 轮快照 | 16,225 | 1.0256 | 0.0338 | 1.4814 |
| E3，275 轮快照 | 16,225 | 1.4086 | 0.6771 | 1.6310 |
| P10，425 轮快照 | 25,075 | 2.2077 | 1.7480 | 2.1189 |
| P15，425 轮快照 | 25,075 | 2.5150 | 2.2651 | 2.3007 |
| P50，425 轮快照 | 25,075 | 2.2612 | 1.9960 | 1.9132 |
| E4，425 轮快照 | 25,075 | 2.3167 | 2.0913 | 2.1929 |

这些快照的完整逐声部指标和最终轮结果见 [实验报告](docs/EXPERIMENT_REPORT.md)。数值来自保存的 JSON 复算，本次未重新使用权重与整曲音频生成指标。P15 在这组已观察快照中的均值较高，不能据此认定其概率最优、干预机制得到验证，或已证明独立跨作品/真实演出泛化。后续应固定验证集选型规则、统一运行协议、增加随机种子，并采用新的独立测试数据。

## 复现与引用

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) 说明可复算与尚未验证的层次；[实验设计](docs/E1-4_DESIGN.md) 记录历史设置；[审核记录](docs/AUDIT_REPORT.md) 说明修订依据。

项目代码沿用原待发布版 MIT 许可证。第三方数据和工具不随代码许可证自动授权，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。若使用原论文或数据，请分别引用其原作者。
