# 子集带背景NNLS组分数量判定

七类候选库，允许0–7个组分，穷举128个子集；不用Top-2或绝对系数分类阈值。

每个子集拟合x=R_S c+B beta，c非负，背景系数不约束符号。背景为谱点索引上的Legendre多项式，候选无背景、0、1、2、3阶。

标准BIC：n log(RSS/n)+k log(n)，n=1024，k为活动材料数加背景参数数。未知噪声方差额外参数对所有模型相同。公式参考[官方统计模型文档](https://www.statsmodels.org/v0.12.2/generated/statsmodels.regression.linear_model.OLSResults.html)。相关光谱点和非负边界使经典渐近保证未必适用。

补充校准BIC把复杂度项乘lambda，候选[0.25,0.5,1,2,4,8,16,32]，不称为标准BIC。

两个规则分别仅以当前RGEP的1344条合成验证谱选择配置：完整集合准确率最高，平局Macro-F1最高，再平局lambda的对数距离1最近、背景阶数最低。标准BIC固定lambda=1。selection.json写入后才读取合成测试及实测数据；不按实测表现扩大搜索。

实现先消去背景，再在七维QR空间枚举子集。NNLS边界最优解由已枚举的较小活动支持集表示，故只需比较所有非负内部解。1e-10倍输入L2的活动系数容差是数值零容差；RSS相对下限1e-14防止精确拟合取对数溢出。

测试与逐子集SciPy NNLS比对，另检验零、单、三组分、背景恢复和尺度不变性。正式数据加载及数组SHA在原Linux容器执行，不跳过指纹不一致。

容器/workspace运行：

~~~bash
python -m unittest tests.test_subset_background_bic
python -u run_subset_bic_comparison.py --output run/SubsetBackgroundBIC_seed19580920_20260925
~~~

两套输入每套56,700条；报告完整集合准确率、数量准确率、Macro-F1及少报/多报，并与冻结RGEP阈值结果做63代理组配对区间。实测标签、组合、行号、源文件及归一化输入SHA均对齐缓存。

验证/实测只有二元真值，校准可能间接偏好二元输出，仍需未来单组分和多组分验证。无需训练网络，不改旧预测和旧阈值。
