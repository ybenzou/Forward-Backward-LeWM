# HAS 论文第一版投稿稿：详细修改计划

## 1. 任务目标与执行约定

目标：基于当前代码、已有结果和英文草稿，完成一版研究主线清晰、论证完整、语言成熟、符合 ICLR 2027 格式的 HAS 投稿稿。优先完成论文叙事和组织上的重构，并接入正在进行的 3 个训练种子 × 3 个任务实验。

本文件是后续执行模型的修改说明，不是论文正文。计划中的标题、表格和检查项便于执行；论文正文应采用连贯的学术段落，不能照搬本计划的任务清单形式。

### 1.1 修改范围

- 论文只围绕 Horizon-Aligned Scoring（HAS）展开。主要比较对象为现有 LeWM 评分和现有长规划跨度对照。
- 正文编辑入口：`paper/iclr2027_conference.tex`。`paper/main.tex` 继续只负责载入正文。
- 现有表格片段可以直接 `\input`；新训练种子表在正文入口内编写，避免增加不必要的文件。
- `paper/iclr2027_conference.bib` 仅在核实引用后做必要更正；已有官方样式、`math_commands.tex`、其他模板配套文件保持原样。符号定义优先在正文入口内完成。
- 保留代码、配置、训练任务、checkpoint、原始数据和已有结果。写作任务不启动训练、评估或新一轮消融。
- 不提交或推送 Git。完成后交付论文改动和简短交接说明，供作者审阅。
- Backward、fusion、TRM 实现、branch-preserving 等仓库探索不进入方法贡献或结果主线。TRM 等相关文献仍可用于研究定位。
- 与 HAS checkpoint 来源有关的辅助训练头，只在实现附录保留一小段必要说明，不展开实验史。

### 1.2 完成标准

读者应能从正文回答：短期规划的评分问题是什么；HAS 怎样处理；Forward 如何训练；哪些实验支持方法优势；动态深度起什么作用；方法在哪些条件下有效。

第一轮执行应完成所有不依赖新结果的写作工作。新结果尚未到齐时，生成可编译、没有虚构数字的完整草稿，待办只保留在 TeX 注释及交接说明中。新结果补齐后再执行第 7 节的接入流程。

“第一版投稿稿”表示文字、结构和证据呈现达到可审阅的投稿稿状态，不表示新增实验已经完成或原始统计已经独立复核。

## 2. 必须落实的写作标准

### 2.1 正向陈述方法

每个设计先说它做什么、怎样工作、带来什么作用。避免以防御性否定解释设计。

| 避免写法 | 推荐方向 |
|---|---|
| “We do X to avoid Y.” | “X provides … through …” |
| “Our method does not generate actions or learn a new metric.” | “HAS scores each candidate through a learned temporal extension of its endpoint.” |
| “These frames are examples, not additional quantitative results.” | 在图注中直接交代轨迹数量、配对起点和任务条件。 |
| “We do not claim a full step-versus-roll sweep.” | 直接给出实际训练目标；删除与未开展工作有关的辩解。 |
| “The official components are unchanged.” 反复出现 | 首次定义共享骨干与求解器，后续直接讨论 HAS。 |

实验协议中的必要区别仍要准确表达，例如 evaluation groups 与 training seeds 的区别。把这类事实放在协议或图注中一次讲清，避免在摘要、引言、结果、结论重复声明。

### 2.2 段落服务论点

每节具有明确的问题和结论。每个主要论证段落按以下顺序组织：

1. 用完整句子提出本段要建立的观点或研究问题。
2. 给出设计依据、机制解释或实验比较。
3. 引用最有代表性的证据。
4. 说明证据意味着什么，并自然引出下一步。

这是一种论证顺序，不是机械模板。不要让所有段落都使用 “To investigate … / We conduct … / Results show …” 开头。背景段不需要强行加入实验结果。

### 2.3 图表支撑结论

- 先提出要检验的优势，再解释比较设计，最后引用图表。
- 一个结果段落通常选择一至两个代表性比较，其余数值留给表格。
- 避免按 PushT、TwoRoom、Reacher 逐项复述整张表。
- 避免以 “Figure X shows …” 或 “Table Y reports …” 作为连续段落的开场。
- 图注负责说明读图必需的信息；正文负责研究解释。颜色、位置、曲线形状不构成正文论点。
- 方法图也应服务研究机制，不写成粉色模块、蓝色箭头、红色弧线的逐项讲解。

### 2.4 有力度且有依据的结论

现有主表支持直接使用：

- “HAS improves success across all three tasks at every evaluated offset beyond the default planning span.”
- “The gains are particularly pronounced in TwoRoom.”
- “The evaluated longer-span LeWM configuration remains below HAS at all offsets of 50 steps or more.”

避免擅自升级为：universally optimal、state of the art、guaranteed reachability、accurate long-horizon prediction、proved causal mechanism、compute-matched efficiency advantage。

不因缺少显著性检验而把已有明确平均提升一律写成 “may help”。只有使用 “statistically significant” 时才需要相应检验。一个清楚的适用条件通常足够，无须在每个结论后追加多句保留意见。

### 2.5 连贯的相关工作

Related Work 使用四个有承接关系的自然段，不使用 `\paragraph{}`、加粗主题标签、项目列表或一篇文献一个小标题。每段围绕一条研究思路组织引用，末句将问题推进到下一段。

### 2.6 文体统一

- 论文用英文；本计划和交接说明可用中文。
- 方法与普遍性质主要用现在时；具体实施的实验按语境使用过去时或现在时，保持一致。
- 统一使用 HAS、Forward imaginer、action block、planning span、goal offset、evaluation group、training seed。
- 删除面向开发者的正文措辞：legacy、v1/v2、pipeline status、protected checkpoint、paper checkpoint 等。具体文件路径只在必要的复现材料中出现。
- “causal” 如无额外定义，仅指前向时间预测或原骨干的因果掩码；不能引出因果识别方面的主张。本文统一优先称 Forward imaginer。

## 3. 核心论点与证据分工

建议采用的中心论点：

> Short-horizon world-model planning benefits from scoring candidate endpoints over the remaining temporal gap to the goal. HAS learns an action-free latent transition and applies it recursively after each action-conditioned rollout, providing a longer-horizon scoring view for the current action sequence.

中文含义：短期动作预测可以通过额外的时间延伸获得更有用的长期评分视角；HAS 用局部训练的 Forward 提供这种评分延伸。

围绕中心论点组织五个层次：

| 论证层次 | 要回答的问题 | 使用证据 | 正文位置 |
|---|---|---|---|
| 问题与方法 | 短期 endpoint 怎样对应较远目标？ | 时间定义、HAS 公式、概念图 | Introduction、Setup、Method |
| 实际收益 | 评分延伸能否改善长 offset 控制？ | 现有十组评估主表与曲线 | Experiments 5.2 |
| 替代方案 | 现有更长规划跨度能否获得类似表现？ | LeWM H=50 主表行 | Experiments 5.2 后半 |
| 设计依据 | 深度安排为什么影响评分效果？ | 固定 k 多组表、k=0 性质 | Experiments 5.3 |
| 稳定性与解释 | 结果跨训练运行如何变化、如何体现于行为？ | 新训练种子、过程轨迹、距离诊断 | Experiments 5.4、5.5 与附录 |

论文以“有效的评分设计”为主要实证结论，以 temporal alignment 为方法组织原则及机制解释。不要另起一条需要新理论或新实验才能闭合的因果证明主线。

## 4. 固定章节结构与篇幅安排

标题保留：`Horizon-Aligned Scoring: A Longer View for Short-Horizon World-Model Planning`。仅根据实际排版决定是否保留人工换行。

正文结构固定如下：

1. Introduction
2. Related Work
3. Problem Setup
4. Horizon-Aligned Scoring
   - 4.1 Scoring beyond the planning span
   - 4.2 Learning the Forward imaginer
   - 4.3 Receding-horizon integration
5. Experiments
   - 5.1 Experimental setup
   - 5.2 Long-offset goal reaching
   - 5.3 The role of alignment depth
   - 5.4 Robustness across training runs（结果完整后启用）
   - 5.5 Planning behavior（5.4 未启用时自动顺延编号）
   - 5.6 Limitations（同上）
6. Conclusion

保留少量有功能的 subsection；删除 Setup、Related Work 等处密集的 `\paragraph{}`。Experiments 的子节按论证问题划分。

正文含图表的目标篇幅：摘要和引言约 1.5 页；Related Work 约 0.7 页；Setup 约 0.6 页；Method 约 1.8 页；Experiments 约 3.8 页；Conclusion 约 0.3 页。总计约 8.7 页，给浮动体留少量余量。这些是编辑预算，不是硬凑页数要求。

附录顺序：A. Training and implementation；B. Evaluation protocol；C. Additional analysis。HAS 训练图、完整评分诊断、新训练种子逐运行表放入对应附录。

## 5. 逐节、逐段修改说明

### 5.1 Abstract：先讲问题与方法，再给最有力证据

目标长度约 170–210 个英文词，最终服从排版与信息完整性。写成一个段落，六至七句。

1. 第一、二句建立任务和瓶颈：latent planner 用短期预测终点给候选动作评分，较远的视觉目标需要更长时间尺度的进展判断。
2. 第三句命名 HAS，并明确“Forward 延伸候选 endpoint 后计算目标距离”。
3. 第四句交代局部训练与递归使用，以及深度随剩余时间变化。
4. 第五句概括三个任务、十组评估中的 long-offset 提升，不在摘要罗列 b、h、H 的全部工程参数。
5. 第六句保留最多两个有代表性的提升：PushT o=50 的 42.8%→67.8%，TwoRoom o=100 的 16.4%→70.4%。这些数字归属现有主实验。
6. 末句提炼评分延伸的意义，可引用 H=50 对照的结果。

删除反复介绍 official setup 的开场。将 “critical bottleneck” 调整为有力度且直接的 “a limitation of terminal scoring” 或等价表达，不附加长篇辩解。

新训练种子结果到齐前不在摘要宣称跨训练种子稳定性。到齐后按第 7 节将摘要关键数字更新为 official/HAS 三训练种子比较；结果支持时加入跨运行一致性的简短从句。

### 5.2 Introduction：五个自然段完成研究动机

**段 1：从任务价值引出评分的重要性。** 用视觉目标控制建立场景，再指出 planner 的成功取决于如何比较候选结果。保留 LeWM/JEPA 等必要引用，避免泛泛介绍整个 world-model 领域。

**段 2：把问题收束到长目标与短规划的时间差。** 解释短期动作有时先完成中间进展，单次 endpoint distance 可能低估其作用。可用 TwoRoom 门口/跨房间情境作直观动机，不把一个成功案例写成普遍定理。本段末提出：候选 endpoint 需要更长时间尺度的评分视角。

**段 3：提出 HAS 并说明关键机制。** 从短 rollout endpoint 到 F 的递归再到目标距离，形成完整动作评分过程。用一句话说明 F 学习数据中的局部 latent transitions，用一句话说明动态深度。概念图在本段附近引用。

**段 4：概括检验逻辑和主要发现。** 先提出检验 long-offset 控制与 depth selection，再给最多两个代表性结果；H=50 用一句话建立比较。新种子仅在可用时加入稳定性一句。

**段 5：用连贯文字收束贡献。** 用三句陈述问题视角、评分方法、实验支持，代替当前 enumerate。每一句包含具体对象和动作，避免 “a novel framework / extensive experiments / comprehensive analysis” 的空泛表述。末句引导读者进入相关研究。

引言只简短交代默认 H=25；完整 b、h、o、e 定义留给 Setup。删除引言中多次重复的 k=0、预测误差、latent geometry 限制。

### 5.3 Related Work：四段连贯的研究推进

目标约 450–600 个英文词；最终允许为主结果腾出篇幅。

**段 1：latent prediction 如何成为规划接口。** 从 latent dynamics planning 过渡到 reward-free / reconstruction-free visual planning，将 PlaNet、Dreamer、TD-MPC 作为背景，再把 PLDM、DINO-WM、LeWM 放入更接近当前任务的研究链。不同工作使用的目标函数有差异，不统一声称它们都采用本文同一种 terminal distance。末句收束到 goal-conditioned latent scoring 的时间跨度问题。

**段 2：长时域方法如何增加规划的时间覆盖。** 将 variable-length dynamics、hierarchical prediction、latent subgoals 和 subgoal-conditioned generation 按作用方式串联，而非逐篇介绍。默认候选引用：VLWM、Hierarchical Planning with Latent World Models、FF-JEPA、SAGE。末句引出在候选序列已经生成后改善评价的问题。

**段 3：从预测表征转向规划评分。** 围绕 planning-oriented representation、trajectory cost 和 reachability 组织 Temporal Straightening、RC-aux、Traj-LeWM、TRM。写清它们处理的评分或表征问题，以及剩余时间为本文提供的切入点。

**段 4：定位 HAS。** 用正向表述概括 HAS 在 action-conditioned endpoint 之后学习时间延伸，并根据剩余 gap 使用同一 Forward。引用最接近的两至三篇工作即可，不重复整个文献清单。本段为 Setup 中的时间定义作铺垫。

引用处理：当前 BibTeX 条目是候选来源，不能只根据标题或旧稿断言方法细节。改写前核对所需文献的原文或官方摘要，记录核查结果到交接说明；不扩展为全面文献调查。无法核实的条目暂不承担具体比较主张。不要为凑文献数新增不相关论文，也不要把未实验比较的文献写成被 HAS 超越的基线。

### 5.4 Problem Setup：一次性建立所有时间尺度

采用三个自然段和必要公式，不设段落小标题。

1. 定义图像 x_t、编码 z_t=E(x_t)、动作块 a、block size b、规划块数 h、环境步跨度 H=hb。当前 P 使用近期 latent/action context，简写 P^(h) 时明确这是省略上下文的记号。
2. 定义初始观察对应的目标 offset o、已执行步数 e、目标 z_g，以及 LeWM endpoint 和 terminal cost。
3. 说明 o-e>H 时剩余 gap 为 o-e-H，并引出“用这个 gap 决定 endpoint 的评分延伸”。交互预算 2o 留在实验协议，避免打断方法动机。

保留原有 `eq:predict-endpoint`、`eq:lewm-cost` 标签。令时间轴相对初始观察定义，避免把绝对 tau 与相对 e 混用。

本节无需重讲 LeWM 的网络结构、训练配置或 CEM 更新步骤。

### 5.5 Method：先给评分机制，再解释它如何学到

#### 4.1 Scoring beyond the planning span

第一段直接定义 F 的单步时间含义、k(e,o) 和 C_HAS。保留 `eq:depth`、`eq:has-cost`，先让读者理解整个方法，再进入训练细节。

第二段解释候选动作通过 P 决定 endpoint，而 F 用轨迹中学到的局部演化规律提供时间延伸。使用 “temporally extended representation” 或 “horizon-aligned scoring representation”，不把它描述为保证可执行的真实未来。

最后用两三句说明动态更新与 k=0 性质，o=100 对应 15→10→5→0 的例子只在此出现一次。附录可保留数值索引表，其他正文不再重复该序列。

技术处理：k 的表达式带 max；当 e+H≥o 时，k=0 的 endpoint 仍位于 e+H，不要在算法中无条件把它标成 tilde z_o。算法统一写 tilde z=F^k(hat z)。

#### 4.2 Learning the Forward imaginer

**段 1：先讲监督来源。** 四帧 z0…z3 相隔一个 action block，p_i 对应 z_(i+1)。说明 LeWM 提供已编码的局部转移训练接口。LeWM 训练目标用一个紧凑公式介绍，网络参数细节移入附录。

**段 2：给出两个单步对与一个递归对。** 明确 F(p0)→z2、F(p1)→z3、F(F(p0))→z3。解释单步项学习局部演化，递归项直接训练复合使用；这就是设计依据，无需补写未做的 loss ablation。

**段 3：交代 detach 和模型形态。** 输入和 target 都 detach；Forward loss 的直接梯度只进入 F。正文只用一句话说明小型残差 MLP；192/768、LayerNorm、GELU 等完整结构在附录。段尾衔接 F 的递归使用与规划集成。

统一损失定义，修正当前公式与代码 mean reduction 的区别：先定义 D(z,z')=||z-z'||²/d，训练式使用 D。

- L_LeWM=(1/3) Σ_(i=0)^2 D(p_i,z_(i+1))+0.09 L_SIGReg。
- L_step=(1/2)[D(F(bar p0),bar z2)+D(F(bar p1),bar z3)]。
- L_roll=D(F(F(bar p0)),bar z3)。
- L_F=L_step+L_roll，Forward 总权重为 1。
- 规划成本继续使用 ||·||²，与现有实现一致。

保留相关公式标签，避免重新编号造成断链。归一化是一处公式修正，不在正文展开成争论。

#### 4.3 Receding-horizon integration

用一段说明：CEM 生成动作块候选、P 预测 endpoint、F 延伸并评分、精英候选更新分布、执行 prefix、更新 e 后重规划。

保留一个紧凑 Algorithm，使用输入、深度、rollout、score、elite update、return 的逻辑。算法与正文不重复逐行解释 CEM。N=300、M=30、J=30 放入协议，不必塞入方法动机。

结尾只陈述计算形式：每个候选增加 k 次小型 MLP 前向；动作候选序列仍长 h。没有计时数据时不写 wall-clock speedup 或 fixed-compute superiority。

### 5.6 Experiments：围绕三个核心问题组织主要结果

#### 5.1 Experimental setup

开场一句概括检验目标：long-offset 控制收益、alignment depth 和训练运行稳定性。

用两至三个自然段交代：

- PushT、TwoRoom、Reacher；offset 25/50/75/100；每组 50 episode；十个 evaluation groups 42–51；相同 task/group 的方法复用 starts；交互预算 2o。
- 现有主对照使用每任务一个相同 checkpoint 上的 LeWM H=25、HAS H=25、LeWM H=50。将 LeWM H=25 定义为共享 checkpoint 的基线评分配置，不暗示额外训练了官方独立模型。
- 共享的 CEM 数量配置；H=50 对照同时执行 50 步再规划。直接称它为 longer-span configuration，在此一次说明 receding horizon，后续无需反复添加限制。
- 主表 mean±std 的单位为十组成功率。仅保留实际可以追溯的统计区间；没有完整统计表时，删除“全文报告 paired CI”的笼统承诺。

完整训练超参数、成功判定细节和 manifest 协议放附录。未核实的环境阈值不能补写。

#### 5.2 Long-offset goal reaching

**段 1：建立主要优势。** 主题句先陈述 HAS 在全部已测试 long-offset 条件下改善目标到达。主表和曲线作为证据，选 TwoRoom o=100 和 PushT o=50 两个具体比较说明收益规模。不要把三个任务的所有 offset 按顺序抄出来。

**段 2：解释跨任务表现。** 讨论不同控制情境中，延伸后的 endpoint scoring 如何与中间进展相联系。TwoRoom 的提升最大、PushT 长 offset 仍困难、Reacher 高基线下仍有提升。把行为机制表述为与观察一致的解释，避免声称实验已经单独识别唯一原因。

**段 3：回答更长动作规划这一替代方案。** 说明此比较检验已有 longer-span configuration；指出 o≥50 时该配置均低于 HAS。可选择 TwoRoom o=100 的 H=50 23.0% 与 HAS 70.4% 作例子。得出的结论是 HAS 的评分设计具有实际价值，不写“长 horizon 必须失败”或“已排除所有其他机制”。

o=25 的 H=50 overshoot 条件留在表注与协议，不在主结果段重复讨论。

#### 5.3 The role of alignment depth

**段 1：先提出设计问题。** Forward 提供评分延伸，而递归深度决定延伸到何处；比较动态规则与固定 k 检验深度选择的作用。直接引用现成 `paper/tables/tab_k_depth_multiseed.tex`，保留其数值与列顺序，新增表标签 `tab:k-depth`。

**段 2：总结最主要模式。** 动态 HAS 在表中六个 task/offset 条件中的五个取得最高平均成功率；固定较深递归在 PushT 尤其不利。用这一模式支持“remaining-gap schedule 是跨条件表现较强的默认选择”。挑一个比较即可，不把六行逐一解释。

**段 3：简洁交代例外并收束。** TwoRoom o=100，固定 k=5 为 76.2±5.7，HAS 为 70.4±7.2。用一句陈述这一条件特征，并指出有效深度也受到任务演化与递归预测误差影响；不要写成已证明的原因。动态规则的优势是整体适用表现，不能称为逐任务最优。

固定 k 的具体语义必须在表注或协议交代：每次 replanning 都使用同一深度，包括动态深度已为零的后期。此实验检验整套深度调度，不能单独分离“早期精确对齐”与“后期停止延伸”的贡献。

k=0 的评分恒等性质用一句引用方法部分，并说明 o=25 的观测表现接近。保留现有主表 87.2/87.0 的 PushT 差异；不自行归因为浮点、随机数或 GPU 非确定性，也不人为改成相同数字。

#### 5.4 Robustness across training runs

按第 7 节的确定流程接入。若完整结果尚未提供，本节用 TeX 注释预留插入点，编译 PDF 中不出现空节、占位表或 “experiments are running”。继续完成其他章节。

#### 5.5 Planning behavior

用一至两个段落建立行为层面的解释。主题句围绕“有用的当前动作可能表现为中间进展”。随后引用配对轨迹，选择 TwoRoom 的通道进展和 PushT 的接触准备说明这种行为；Reacher 只作简短补充。

只写现有轨迹和原始说明实际支持的动作，不凭单帧推断决策意图或虚构行为。若无法查看 PDF 或轨迹，不新增具体动作描述，保留可确认内容。

评分距离诊断放附录，正文最多用两句承接：考察 Forward 后 latent 与远期目标的距离变化，为评分表示提供补充观察。该诊断以 encoder endpoint 为输入且没有 CEM candidate ranking，不能当作排序准确性测试。使用 “evaluation-disjoint windows” 表达已有的 episode 排除规则；只有另有训练划分证据时才称 training-held-out。

#### 5.6 Limitations

写一个紧凑段落，默认约 100–150 个英文词。集中讨论已知 goal offset 的使用、action-free recursion 在长距离或多模态转移中的局限，以及评估范围。

新训练种子完整后，说明主要方法比较覆盖三个训练运行，替换“只评估一个训练 checkpoint”的笼统句子；将仍基于单 checkpoint 的 H=50 对照和深度消融与主要比较区分清楚。

删除正文其他位置重复出现的局限陈述。无需将所有未完成基线、未做消融、历史探索列入此节。

### 5.7 Conclusion：回到研究观点

一个自然段，四至五句，约 100–130 个英文词。

1. 重述短期预测 endpoint 的长期评分问题。
2. 概括 HAS 的局部训练、递归延伸和动态深度。
3. 总结三个任务的 long-offset 收益。
4. 用一句话提炼研究意义：评分的时间尺度是改善短期 world-model planning 的有效设计维度。

不再列具体数字、checkpoint 数量或所有对照结果；不重复 Limitations，不增加新贡献。

## 6. 图表、附录与旧内容迁移

### 6.1 固定的主文图表配置

| 元素 | 位置 | 要支撑的论点 | 修改要求 |
|---|---|---|---|
| `fig:overview` | Introduction 方法概述附近 | 从短 endpoint 到更长评分视角 | 保留 PDF，图注改为机制说明，删除颜色清单。 |
| `fig:multiseed` | 5.2 或附录 | long-offset 下的整体收益趋势 | 保留现有 PDF；明确这是原单 checkpoint 的十组评估，不能改图注冒充三训练种子结果。新主表接入后优先移入附录，避免统计口径混淆。 |
| `tab:main` | 5.2 | HAS 相对 LeWM 的控制收益 | 新结果到齐前保留原表；到齐后更新为 official/HAS 三训练种子表，旧三方比较完整迁移为独立 H=50 对照表。 |
| `tab:k-depth` | 5.3 | 深度调度的作用 | 引用已有十组表；定义固定深度、动态 HAS、统计单位。 |
| 三训练种子逐运行明细 | 附录，5.4 引用 | HAS 相对 official 的跨运行收益一致性 | 主文复用更新后的 `tab:main`；附录按 task/train seed/method 列出四个 offset，不另设重复的 HAS-only 表。 |
| `fig:process` | 5.5 | 中间进展与最终到达的行为例证 | 保留 PDF；图注明确配对起点、offset 和时间间隔。 |

主结果曲线和主表分工不同：曲线表达趋势，表格给出精确对照；正文不重复两遍解读。

### 6.2 附录配置

- 将 `fig:method` 训练图及完整架构放入 A，正文 4.2 引用。关键训练公式保留正文。
- B 收录 CEM、执行长度、interaction budget、starts、统计方法和具体环境 success 定义。
- C 收录 `fig:score`、其采样规则及诊断解释；新训练种子完整后加入每个运行的成功率明细。
- 删除旧的独立 “Action-conditioned Forward check” 章节、`tab:aaf` 和其正文引用。该小规模探索不是本版 HAS 论证的必要组成，且不同来源 seed-42 数字尚未统一。
- `tab_k_depth.tex` 是旧单组片段，不用于本版主张；文件保留，不混入多组统计。
- 辅助 Backward 头在 A 仅保留三句左右：checkpoint 包含参数独立的辅助头；输入与目标 detach；HAS 评估采用 Forward 路径。不要扩展为“加入/删除该头完全不影响训练轨迹”的主张。

### 6.3 图像检查与排版处理

第一轮沿用现有 PDF，不重新生成实验图，不要求接入服务器。移动图位置、缩放图宽和修改 caption 均在 TeX 中完成。

默认将现有 `[H]` 改为 `[t]` 或 `[tbp]`，取消包围这些图的局部负担较大的间距强制设置。删除不再需要的 section-level float barrier，使 LaTeX 正常安排浮动体。不得通过修改官方 margin、字体或全局行距挤进页数。

需要压缩正文时按以下顺序处理：删重复句和完整表格复述；精简 caption；将 process 图移到附录、保留正文分析与引用；压缩 Related Work 次要背景引用。核心 HAS 公式、主结果表和固定深度表优先保留正文。

## 7. 3 个训练种子 × 3 个任务：结果接入规则

### 7.1 已知实验设计与默认定位

作者告知该实验正在进行。仓库中的 `scripts/run_forward_train_seeds.py` 默认训练种子为 3072、3073、3074，任务为 PushT、TwoRoom、Reacher，默认评估组 42–51，offset 25/50/75/100，每组 50 episode。

作者已确认实际实验设计：3 个训练种子 × 3 个任务，每个运行均包含 official（LeWM）与 HAS 两组评估，覆盖 offset 25/50/75/100。以这一实际安排为准，不能根据单个启动脚本的 `--modes=forward` 默认参数将实验误判为 HAS-only。新实验同时用于检验跨训练运行的性能和相对 LeWM 的收益。

结果完整后，将三训练种子的 official/HAS 成对比较作为主要效果证据，统一更新摘要、引言、结果与结论中的核心数字。优先把 `tab:main` 更新为这两组的三训练种子汇总，将现有单 checkpoint 的 H=25/H=50/HAS 三方比较保留为独立的长跨度对照表。不得将旧 H=50 行与新三训练种子两行混成同一统计口径。第 5、6 节要求保留的旧数值保留在该对照表中；相关段落随此调整，避免重复设置内容相同的主结果表与稳定性表。

脚本支持复用 s3072 的旧 checkpoint。实际是否复用、实际种子数和评估组数以实验产物为准，不单凭脚本默认参数断言。复用同一 checkpoint 不计为一个额外独立训练运行。

### 7.2 读取与完整性检查

预期结果目录为 `outputs/diag/train_seeds/{task}/s{train_seed}/seed_{eval_seed}/`。实现者只读取已有结果；本地缺失时继续完成写作并在交接说明中列出需要作者提供的目录或汇总。

接入前检查：

1. 三个任务各有三个不同训练运行，记录 checkpoint 路径和实际训练配置。
2. 每个运行均包含 official（LeWM）和 HAS 的预期评估组及四个 offset；核对每组 episode 数和完成状态。
3. 复用的是指定 starts manifest；没有误混其他 variant、不同训练 epoch 或额外 protocol。
4. 任务/训练 seed/方法/评估 seed/offset 五元组没有重复或缺项；同一运行两种方法的起点、目标与评估预算一致，记录实际 checkpoint 对应关系。

若实际沿用十组评估，则设计共 3×3×2×10×4=720 个成功率单元，即 360 对方法比较。它们不是 720 个独立训练重复，也不意味着所有 episode 起点彼此不同。

### 7.3 确定的统计方法

对每个 task 和 offset，令 r_(s,g,m) 为训练运行 s、评估组 g、方法 m（official 或 HAS）的成功率，单位为百分比。

1. 先分别按训练运行和方法计算 R_(s,m)=(1/10)Σ_g r_(s,g,m)。
2. 对每种方法报告三个 R_(s,m) 的均值，以及样本标准差 `std(ddof=1)`。
3. 更新后的主表每个 task 包含 official（LeWM）和 HAS 两行，四列对应 offset；每格为上述跨训练运行 mean±std。不再另外放一张重复的 HAS-only 稳定性表。
4. 按相同训练运行计算 Δ_s=R_(s,HAS)−R_(s,official)，报告三个 Δ_s 的平均提升（百分点）；需要表示提升的波动时计算 Δ_s 自身的样本标准差，不用两组标准差相减。附录列出 task、train seed、方法和四个 R_(s,m)，便于核查每个运行的提升；caption 说明每个 R_(s,m) 汇总十组评估。
5. 不把 30 个评估组当作 30 个训练 seed，不对三个运行中的最佳结果挑选报告。

若实际运行参数不同，按实际的完整评估组进行对应汇总，并如实修改协议；不同运行覆盖不同组时先报告不完整，不用不均衡数据自动生成三运行总表。

### 7.4 结果段落如何写

段首先提出收益的一致性问题：HAS 相对 official（LeWM）的控制收益是否跨训练运行保持。第二句说明三个训练种子、双方法评估与共享评估条件。随后根据实际配对提升和训练运行间离散程度概括观察，并引用新主表。主要效果段落承担平均收益的论证，本段集中讨论跨运行一致性，不再重复罗列数字。

训练间 SD 较小且结果接近时，可写 “HAS exhibits consistent performance across three training runs.” 某任务波动较大时，直接说明稳定性在任务间的差异，不预先指定它一定稳定。

完整结果支持时，可以直接写 HAS 的收益跨训练种子保持一致。仅当所述 task/offset 范围内每个训练种子的 Δ_s 均为正，才写 “HAS outperforms LeWM across all three training seeds”，并明确范围；只在三种子均值上领先时，就陈述平均收益。offset=25 单独检查其接近基线的表现，不预先要求或声称四个 offset 全面提升。

### 7.5 未完成结果的处理

结果到齐前，仅在正文入口保留一个清楚的 TeX 注释块，例如：

```tex
% HAS_TRAIN_SEEDS_PENDING:
% Integrate official (LeWM) and HAS results for 3 train seeds x 3 tasks x 4 offsets.
% Aggregate evaluation groups within each training run, then mean/std across runs.
% Update the main paired comparison and report within-run HAS-minus-official gains.
% Keep the existing single-checkpoint H=50 comparison in a separate control table.
```

不输出假数字、预计结果、TBD 表格或实验进行中的论文段落。摘要和结论依据已经完成的实验写成完整文字。

## 8. ICLR 2027 格式与声明

已于 2026-09-09 核查官方指南：初稿正文至多 9 页；参考文献与附录不计入正文限额，附录置于参考文献之后；投稿采用匿名格式。AI use statement 为必需项且不计页数，Ethics 与 Reproducibility statement 为推荐项并按指南处理。使用官方样式。[ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)

保留当前项目提供的 ICLR 2027 模板结构与 `%\iclrfinalcopy` 注释状态；核对输出 PDF 的匿名标题与页眉。不要因文件名包含 2027 就跳过实际格式检查，也不自行更改官方样式。

AI use statement 保留真实的实施辅助、结构规划、起草与编辑使用情况，具体披露按官方政策核对。删除当前模板中写给作者的括号提示语，PDF 只出现正式声明内容。[ICLR 2027 AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors)

Ethics statement 用简洁事实描述模拟控制的研究范围。Reproducibility statement 指向方法、协议、附录和实际准备好的补充材料；尚未提供的 checkpoint 或代码不要写成已经公开。投稿稿中的链接应符合匿名要求。

本计划不办理 OpenReview 投稿、作者登记或上传材料。

## 9. 实施顺序与每步交付

### 第一步：建立结果与段落的对应关系

阅读 PAPER.md、当前正文、两张 k 表、训练种子启动脚本。以当前正文主表和十组 k 表为第一版数值依据；FIGURES.md 是历史参考，不能覆盖较新的表格。

记录当前 Git 状态并保存需要核对的证据清单到执行过程说明。不要为这一步创建新的研究报告或修改项目文档。

### 第二步：重构 Method 和 Setup

先固定符号、损失定义、评分公式和章节顺序。完成后检查公式与实际 latent Forward 实现一致，确保后续引言能引用清楚的技术对象。

交付：连贯的 Setup、三个 Method 子节、一个正确的算法。

### 第三步：重构实验论证

新结果未到齐时保留主表；到齐后按第 7 节升级为双方法三训练种子比较，并独立保留旧 H=50 对照；接入固定 k 多组表；重写 behavior 分析；落实跨运行收益段落；合并局限；迁移附录。

交付：每个实验小节都有明确问题、比较设计、核心发现和结论，正文不按表逐行复述。

### 第四步：重写 Introduction 与 Related Work

严格按第 5 节的五段引言与四段 Related Work 执行。核查近邻文献之后再完成方法定位。不增添新的实验优越性或未经核查的引用。

交付：研究动机、方法与实验形成一致的故事；Related Work 没有段落小标题。

### 第五步：完成摘要、结论与声明

最后写 Abstract 和 Conclusion，使它们只总结正文已经建立的内容。调整 AI/Ethics/Reproducibility statements，删除模板说明语。

交付：摘要与结论独立可读，术语、数字归属与贡献保持一致。

### 第六步：语言与排版通读

逐段删除防御性否定、空泛名词、重复背景和图表描述。检查每段主题句及跨段衔接，避免把所有段落改成相同句式。

编译后根据实际页面进行有限排版调整；优先精简和移动材料，不压缩官方格式。

### 第七步：接入新训练结果

如结果已经完整，执行第 7 节的统计、表格与文字更新，并重新编译。若仍缺失，交付完整第一轮草稿并准确说明唯一待接入的结果部分，不暂停已可完成的写作。

## 10. 验证与验收清单

### 10.1 内容验收

- [ ] 一句话能说清本文主张，所有主要实验都能对应到该主张的一个问题。
- [ ] 正文聚焦 HAS，没有 Backward/fusion/实验版本史。
- [ ] Introduction、Related Work、Setup、Method、Experiments、Conclusion 存在自然承接。
- [ ] Related Work 为四个自然段，没有 `\paragraph{}` 或加粗的主题标签。
- [ ] 方法解释直接讲实现思路，未出现反复的 “not / rather than / we do not claim” 防御性组织。
- [ ] 每个结果段先讲要建立的结论或问题，再引用必要数字与图表。
- [ ] 主结果至少清楚建立 long-offset 收益、更长跨度对照、深度安排三项论证。
- [ ] 固定 k=5 在 TwoRoom o=100 的例外被简洁交代；没有改写为动态 HAS 全部最优。
- [ ] evaluation groups、training seeds、checkpoints 三个概念及各表统计单位清楚。
- [ ] 未完成的新实验没有被写成已完成，没有空白结果表进入 PDF。
- [ ] 限制集中于一个段落，其他章节没有反复重复免责声明。

### 10.2 技术与数值验收

- [ ] b=5、h=5、H=25、goal offset、elapsed steps 的单位一致。
- [ ] k=0 时算法不错误标注为已经到达 goal time 的预测。
- [ ] F 的监督索引、detach 边界和 mean reduction 与代码一致。
- [ ] H=50 的实际执行 prefix 与规划跨度都写清楚。
- [ ] 原主表数值在保留或迁移后的 H=50 对照表中逐格保持不变；新主表数字来自完整的双方法三训练种子结果。发现来源冲突时标记，不自行修正数字。
- [ ] k 消融使用十组表，不混入历史 seed-42 表。
- [ ] 新训练表按“先组内、再运行间”汇总，SD 使用 ddof=1。
- [ ] 新实验按作者确认的 official/HAS 双方法设计接入；两组统计口径一致，配对提升按训练运行计算，未与旧 H=50 单 checkpoint 结果混合汇总。
- [ ] 评分距离诊断没有被描述成 CEM 排序准确性实验。
- [ ] 每个 citation key 有对应条目；具体文献比较有核查依据。

### 10.3 编译与视觉验收

在 `paper/` 目录优先使用：

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

若没有 latexmk 但有完整 pdflatex/bibtex 环境，使用 pdflatex → bibtex → pdflatex → pdflatex。若本地没有 TeX 环境，不为写作擅自安装大型工具链；完成静态检查，并在交接说明中明确需要 Overleaf 编译。

实际检查：

- 无编译错误、未定义引用、未定义文献或缺失图像。
- 主文计数符合 9 页限制，附录在参考文献后，匿名信息正确。
- 无图表或公式越界；表格字号可读；caption 与图内容一致。
- 无孤立标题、明显浮动空白、正文中断或图堆积到末尾。
- 全文查看导出的 PDF，不仅检查编译退出码；若无法查看，明确说明未完成视觉验收。
- 生成文件只作为本地产物，不把 aux/log/preview 图片加入 Git。
- 结束时检查 diff，确认改动仅涉及本任务允许的论文内容。

### 10.4 最终交接格式

用简短中文说明以下四项即可：

1. 已完成的结构和论证变化。
2. 新训练种子是否接入，以及所用统计单位。
3. 编译、页数和视觉检查的实际结果。
4. 尚需作者提供的具体数据或核实的具体事实。

不以“还需要大量实验”作为默认结尾，不把与第一版投稿稿无关的研究方向列为阻塞项。

## 11. 可直接交给执行模型的任务说明

> 请以本 plan.md 为实施依据，对当前 HAS 英文论文做第一版投稿稿重构。先阅读 PAPER.md 与正文，再依照第 9 节顺序完成修改。论文只讨论 HAS，以论点组织段落，以实验和图表支撑论证；重点落实第 5 节逐段要求。保持已有数值与实验条件，按第 7 节处理尚在进行的三训练种子实验。正文写入 paper/iclr2027_conference.tex，必要时核实并更正文献条目，保持官方模板与实验代码不变。完成所有不依赖新数据的工作，进行可用的编译与视觉检查，并按第 10.4 节交接。不得虚构实验、引文或编译成功状态，不执行训练、评估、Git 提交或推送。
