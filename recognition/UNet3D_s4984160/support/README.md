# 辅助内容

日常先看上一层的五个文件即可：
**dataset.py → modules.py → train.py → evaluate.py → predict.py**。
总报告和对比表统一在 [项目 README](../README.md)。

这里集中保存其余内容，避免挤在主目录：

| 内容 | 用途 |
| --- | --- |
| `baseline_model.py` | 按检查点配置构造实际训练的归一化层 |
| `evaluation_utils.py` | 指标、空间信息和权重加载辅助函数 |
| `prepare_comparison.py` | 核对原始指标并更新总报告对比表 |
| `plot_training_curves.py` | 读取已有 CSV 绘制 3D 训练曲线，不训练模型 |
| `training/` | 初始从头训练阶段；根目录 train.py 是后续训练阶段 |
| `scripts/` | 三个当前 Slurm 提交脚本 |
| `tests/` | 合成数据检查 |
| `docs/` | 方法说明、案例分析、数据对应和评价规则 |
| `artifacts/final_test_3d/` | 原始指标、训练记录和 figures 中的五张图 |
| `comparison/` | 机器可读的对比数据 |
| `history/` | 旧实验和原始源码版本记录，不是当前入口 |

上述移动只整理本地/GitHub。集群已提交任务仍使用自己的冻结快照，未被修改。
