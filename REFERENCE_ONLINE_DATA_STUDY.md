# RGEP训练分布对照与原始谱NNLS背景基线

## 目的与边界

检验训练—验证分布不匹配是否影响弱组分检出和检查点稳定性。不预设深度方法优于NNLS，也不将很低的训练损失单独视为已证明“记忆了所有样本”。

不更换网络、不改变光谱损失、不使用实测混合物更新参数。七类训练参考库、来源隔离清单和旧合成验证/测试集保持不变。

## 三个训练配方

| --training-data | 数据 | 用途 |
|---|---|---|
| original_p | 原来源审计的固定10,752条 | 原配方对照 |
| fixed_simulated | 训练纯谱增强生成一次，之后重复 | 检验比例、噪声及背景分布变化 |
| online_simulated | 相同配方每轮重新生成 | 相对固定增强组检验刷新样本的作用 |

增强直接调用生成验证集的 rssnet_data.simulate：

- 仅使用固定清单的训练纯谱；不抽取验证/测试纯谱。
- 每组合512条、21组合，每轮10,752条，7查询/混合谱。
- 光谱系数0.1–0.9；白噪声与累积随机噪声，SNR 10–30 dB；轻度二次背景；沿用原函数非负截断。
- 输入L2归一化，干净加权端元共用输入缩放因子。背景和噪声不作第三组分目标。
- 按种子、训练专用域、轮次确定合成流，不依赖批次读取顺序。固定与在线增强第0轮相同。
- 查询批量32时，每轮2,352次更新。在线组见到的不同合成谱更多，这是被检验变量。

“在线”在当前实现中指每轮开始刷新内存数据，而不是每次读取同一个批次都重新随机生成，避免Keras预取、重试影响采样审计。

## 选模与审计

主检查点仍按完整合成验证Top-2最高、平衡BCE较低、较早轮次选取。另存同一次运行最低验证联合损失的权重和轮次；不看实测结果在两者间挑更好者。

正式新增合成训练必须提供 --evaluation-anchor。固定纯谱清单、验证/测试数组及参考须与旧运行一致，否则停止。Windows与原Linux环境可能有微小数值差异，正式实验应回到原Docker环境，不删除校验来强行运行。

每轮 result/query_audit.jsonl 记录查询顺序、合成种子、训练数组校验和、样本数及系数范围。固定验证数组另存 result/validation_data.npz，原始谱评价读取并校验，不重新生成替代验证集。

## 先短程检查

容器内 /workspace：

~~~bash
python -u train_reference_extractor.py \
  --training-data online_simulated \
  --checkpoint-selection val_top2 \
  --version 2 --short-check \
  --epochs 2 --batch-size 32 --save-every-epoch \
  --output-root run/ReferenceOnline_short
~~~

短程只用每组合32条训练谱，使用完整固定验证集，不评价合成测试或实测数据。检查 validation_health.json 和 query_audit.jsonl；两轮不足以评价最终效果。health_passed只是健康阈值，不是正式准确率结论。短程权重被禁止用于正式原始谱测试。本地CPU检查加 --cpu。

~~~bash
python -m unittest tests.test_reference_online tests.test_reference_extractor tests.test_reference_checkpoint
~~~

## 正式训练：一个随机种子的三配方对照

~~~bash
python -u run_reference_data_study.py \
  --output run/ReferenceDataStudy_seed19580920 \
  --seeds 19580920 --epochs 50 --batch-size 32
~~~

三个配方顺序训练，各自新目录，不覆盖历史结果。根目录 status.json 显示当前配方，各配方 train.log 保存日志。此命令只训练和评价固定合成测试集，**不会自动打开原始实测测试集评价**。

旧单次RGEP训练约5小时仅为粗略参考；三配方与完整验证的开销应根据首轮重新估计。

若只先训练在线版本：

~~~bash
python -u run_reference_data_study.py \
  --output run/ReferenceDataOnline_seed19580920 \
  --recipes online_simulated \
  --seeds 19580920 --epochs 50 --batch-size 32
~~~

后续用 --seeds 19580920 19580921 19580922 做三个种子，须使用新目录。只跑在线版不能隔离增强分布与刷新效应，论文仍需固定增强组。加 --presence-only 可训练相同配方的仅存在性对照，评价时同样独立使用固定验证集选阈值。

## 原始谱NNLS＋背景对照

评价器为 evaluate_reference_raw_nnls.py。拟合 x ≈ R c + B β：

- R与RGEP共用训练中位数参考，材料系数c非负。
- B为谱点索引上的低阶Legendre多项式，背景系数β不限制正负，是平滑偏置背景，不声称背景处处非负。
- 正交投影消去背景，再做NNLS、回代背景；测试与直接混合约束求解交叉核对。
- 候选为无背景、0、1、2、3阶，仅用合成验证Top-2选择，平局选简单者。无背景也可胜出，不强行保证新增项有收益。
- 普通NNLS、选中背景配置、RGEP分别用同一合成验证集锁定共享阈值。报告Top-2、完整集合、数量准确率、Macro-F1、少报和多报。

从训练根目录 status.json 取得实际模型目录，替换下面的 TRAIN_RUN（占位符，不能原样运行）：

~~~bash
python -u evaluate_reference_raw_nnls.py \
  --run-dir TRAIN_RUN \
  --output run/ReferenceRawNNLS_online_seed19580920 \
  --conditions 1 2 3 4 5 6 7 8 9 --batch-size 32
~~~

评价三配方时使用相同条件和规则，保留全部结果，不按实测分数改方案。每次输出目录必须不存在。

只检查验证选模与阈值流程，不读取实测数据：

~~~bash
python -u evaluate_reference_raw_nnls.py \
  --run-dir TRAIN_RUN \
  --output tmp/ReferenceRawNNLS_validation_check \
  --validate-only --cpu
~~~

原始输入取 dataset/dataset_mixture_raw/dataset*_norm.mat 中**不带_norm后缀的组合变量**。所有模型只共享L2缩放，不调用去噪器或事先扣背景。文件名含_norm不等于网络已去噪；变量存在性、行数、输入校验均检查。它是发布的重构前谱，不能未经核实称为完全未经仪器处理的探测器原始计数。

训练纯谱仍为重构后来源，原始混合谱可能有更大域差异。既定合成增强不保证覆盖实际噪声、荧光和响应变化，结果可能更差。七类参考库与二元真值限制仍存在，本实验不证明开放集、任意组分数或质量丰度预测能力。

## 固定模型的网络去噪前后对照

在同一评价命令中增加 `--input-source paper-denoised`，读取论文发布的
`dataset/dataset_mixture_denoised/dataset*_denoised.mat` 的 `*_norm` 变量，再做共同L2缩放。
这使用已有网络去噪输出，不重新运行本地基础或多质量去噪器。
增加 `--compare-evaluation RAW_EVALUATION` 指向已完成的原始谱结果目录，会核对权重、参考库、
验证数据、阈值、背景阶数及每条谱的组合和行号；任何不一致均停止。
`TRAIN_RUN` 和 `RAW_EVALUATION` 为必须替换的路径占位符。

~~~bash
python -u evaluate_reference_raw_nnls.py \
  --run-dir TRAIN_RUN --output run/ReferenceDenoisedNNLS_new \
  --input-source paper-denoised --compare-evaluation RAW_EVALUATION \
  --conditions 1 2 3 4 5 6 7 8 9 --batch-size 32
python summarize_reference_raw_nnls.py --evaluation run/ReferenceDenoisedNNLS_new
~~~

去噪前后使用相同锁定阈值，用于隔离输入变换的影响，而不是为每种输入单独寻优。
结果包含逐条件指标、逐样本决策和去噪前后对比；此对照不等于有真实端元真值的分离质量评价。
