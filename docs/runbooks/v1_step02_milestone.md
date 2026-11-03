# V1 Step02 执行链开发里程碑

## 状态

这是开发里程碑，不是完整 V1 或 γ 正式实验发布。

已完成一次真实合成闭环：
prepare → seal → deploy → 最终 CPU/RAM probe → reservation →
B2 launcher → Step02 → monitor → collector → 独立业务复验 →
确切容器核对 → resource release。

正式生产政策未启用；生产结果池未写入。
本轮真实验证了已退出容器的释放。
运行中容器的显式 stop 已有模拟测试，但尚未完成真实停止验收。

## 参数与资源语义

- Step02 的 alpha、beta、gamma、affinity_eps 与variant显式冻结。
- 输入必须来自可信内部产物，PKL加载前检查文件hash。
- CPU与当前可用内存进行一次最终准入。
- CPU配额不等于物理核心独占。
- 其他容器资源限制未知只警告，不再推定内存probe失效。
- 磁盘低于50 GiB或空间未知只警告。
- 同组、同主机的本工具reservation继续互斥。
- 资源准入不保证运行期不会OOM或磁盘写满。
- 不确定提交不得盲目重试；不自动删除容器或结果。

## 结果与入池

COLLECTED_ALL表示结构与传输检查通过，不表示生产入池。
BUSINESS_REVALIDATED表示Step02对象不变量复验通过，不证明求解器最优性。

stage_ingest_sandbox.py只允许带明确标记的隔离池。
不得删除sandbox保护或修改标记后直接用于生产池。

本次不提交带历史审计目录硬编码的证据驱动工具：
admission_evidence_v1.py、ingest_stage_evidence.py。
这些工具的验收证据仍保留在仓库外。

## 测试与可移植性

本轮验收：
- wrapper集中回归436项；
- OGD集中回归91项；
- 一次真实合成资源生命周期闭环。

部分wrapper集成测试需要通过环境变量提供已生成的合成fixture：
OGD_C2D1_FIXTURE_INPUT、OGD_C2D5C_ACCEPTANCE。
部分OGD复验/事务测试需要 OGD_REVALIDATION_FIXTURE。

因此，全新clone不能仅靠unittest discover自动复现所有集成测试。
发布前必须把这些fixture生成过程整理为可移植测试设施。
不得将商业资产、完整审计目录、SSH私钥或运行缓存提交到Git。

## 后续放行门槛

1. 测试fixture可移植化；
2. 少量真实γ data2输入的骨架、帧轴、mesh和维度准入；
3. 真实长任务的停止/恢复运维验收；
4. 正式生产入池接口、写锁协调、累计维度与artifact能力验证；
5. Step03/04阶段协议及完整data4保存逐阶段完成。

不要将本次开发提交等同于以上门槛全部通过。
