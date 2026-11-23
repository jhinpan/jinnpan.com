PLATE_DECISIONS = r"""
<svg viewBox="0 0 880 310" role="img" aria-labelledby="p1-t"><title id="p1-t">five decisions and who makes them</title>
<defs>
  <marker id="ah" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#14315c"/></marker>
  <marker id="ahr" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#c2513a"/></marker>
  <marker id="aht" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#1d7d6b"/></marker>
</defs>
<rect class="nd" x="16" y="22" width="300" height="40"/>
<text class="tx b" x="30" y="47" lang="en">work split · grid · tiles · split-K</text>
<text class="tx b" x="30" y="47" lang="zh">工作划分 · grid · tile · split-K</text>
<rect class="nd" x="16" y="76" width="300" height="40"/>
<text class="tx b" x="30" y="101" lang="en">data movement · layouts</text>
<text class="tx b" x="30" y="101" lang="zh">数据搬运 · layout</text>
<rect class="nd" x="16" y="130" width="300" height="40"/>
<text class="tx b" x="30" y="155" lang="en">instruction choice</text>
<text class="tx b" x="30" y="155" lang="zh">指令选择</text>
<rect class="nd" x="16" y="184" width="300" height="40"/>
<text class="tx b" x="30" y="209" lang="en">register use · occupancy</text>
<text class="tx b" x="30" y="209" lang="zh">寄存器使用 · occupancy</text>
<rect class="nd" x="16" y="238" width="300" height="40"/>
<text class="tx b" x="30" y="263" lang="en">instruction order · waits · priority</text>
<text class="tx b" x="30" y="263" lang="zh">指令顺序 · wait · 优先级</text>
<rect class="nd teal" x="400" y="26" width="210" height="104"/>
<text class="tx b" x="505" y="58" text-anchor="middle" lang="en">kernel author</text>
<text class="tx b" x="505" y="58" text-anchor="middle" lang="zh">kernel 作者</text>
<text class="tx" x="505" y="82" text-anchor="middle" lang="en">Python · layout algebra</text>
<text class="tx" x="505" y="82" text-anchor="middle" lang="zh">Python · layout 代数</text>
<text class="tx" x="505" y="104" text-anchor="middle" lang="en">atoms · explicit hints</text>
<text class="tx" x="505" y="104" text-anchor="middle" lang="zh">atom · 显式提示</text>
<rect class="nd red" x="400" y="186" width="210" height="104"/>
<text class="tx b" x="505" y="218" text-anchor="middle" lang="en">LLVM heuristics</text>
<text class="tx b" x="505" y="218" text-anchor="middle" lang="zh">LLVM 启发式规则</text>
<text class="tx" x="505" y="242" text-anchor="middle" lang="en">instruction order</text>
<text class="tx" x="505" y="242" text-anchor="middle" lang="zh">指令顺序</text>
<text class="tx" x="505" y="264" text-anchor="middle" lang="en">registers · waits</text>
<text class="tx" x="505" y="264" text-anchor="middle" lang="zh">寄存器 · wait</text>
<rect class="nd" x="700" y="108" width="164" height="100"/>
<text class="tx b" x="782" y="150" text-anchor="middle" lang="en">final ISA</text>
<text class="tx b" x="782" y="150" text-anchor="middle" lang="zh">最终 ISA</text>
<text class="tx dim" x="782" y="174" text-anchor="middle" lang="en">what the GPU runs</text>
<text class="tx dim" x="782" y="174" text-anchor="middle" lang="zh">GPU 真正执行的</text>
<path class="edge teal" d="M316,42 L398,58" marker-end="url(#aht)"/>
<path class="edge teal" d="M316,96 L398,80" marker-end="url(#aht)"/>
<path class="edge teal" d="M316,150 L398,104" marker-end="url(#aht)"/>
<path class="edge" d="M316,204 L398,224" marker-end="url(#ah)"/>
<path class="edge" d="M316,258 L398,252" marker-end="url(#ah)"/>
<path class="edge" d="M610,78 L698,136" marker-end="url(#ah)"/>
<path class="edge" d="M610,238 L698,182" marker-end="url(#ah)"/>
<path class="edge red" d="M560,130 L560,184" marker-end="url(#ahr)"/>
<text class="tx red" x="550" y="152" text-anchor="end" lang="en">fences · inline asm</text>
<text class="tx red" x="550" y="168" text-anchor="end" lang="en">pins · LLVM patches</text>
<text class="tx red" x="550" y="152" text-anchor="end" lang="zh">fence · inline 汇编</text>
<text class="tx red" x="550" y="168" text-anchor="end" lang="zh">pin · LLVM 补丁</text>
</svg>
"""

PLATE_PIPELINES = r"""
<svg viewBox="0 0 900 380" role="img" aria-labelledby="p2-t"><title id="p2-t">today's pipeline and the redesigned one</title>
<defs>
  <marker id="ah2" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#14315c"/></marker>
  <marker id="ah2r" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#c2513a"/></marker>
</defs>
<text class="lane" x="16" y="20" lang="en">TODAY · FLYDSL 0.4 / 0.5</text>
<text class="lane" x="16" y="20" lang="zh">现在 · FLYDSL 0.4 / 0.5</text>
<rect class="nd" x="16" y="32" width="118" height="56"/><text class="tx b" x="75" y="56" text-anchor="middle">Python</text>
<text class="tx dim" x="75" y="74" text-anchor="middle" lang="en">hidden rule</text><text class="tx dim" x="75" y="74" text-anchor="middle" lang="zh">隐藏规则</text>
<rect class="nd amber" x="148" y="32" width="118" height="56"/><text class="tx b" x="207" y="56" text-anchor="middle">trace</text>
<text class="tx dim" x="207" y="74" text-anchor="middle" lang="en">every call</text><text class="tx dim" x="207" y="74" text-anchor="middle" lang="zh">每次调用</text>
<rect class="nd" x="280" y="32" width="118" height="56"/><text class="tx b" x="339" y="56" text-anchor="middle">Fly MLIR</text>
<text class="tx dim" x="339" y="74" text-anchor="middle" lang="en">layout objects</text><text class="tx dim" x="339" y="74" text-anchor="middle" lang="zh">layout 对象</text>
<rect class="nd" x="412" y="32" width="118" height="56"/>
<text class="tx b" x="471" y="56" text-anchor="middle" lang="en">14 steps</text><text class="tx b" x="471" y="56" text-anchor="middle" lang="zh">14 个步骤</text>
<text class="tx dim" x="471" y="74" text-anchor="middle" lang="en">convert</text><text class="tx dim" x="471" y="74" text-anchor="middle" lang="zh">转换</text>
<rect class="nd" x="544" y="32" width="80" height="56"/><text class="tx b" x="584" y="65" text-anchor="middle">LLVM IR</text>
<rect class="nd red" x="638" y="32" width="152" height="56"/><text class="tx b" x="714" y="56" text-anchor="middle">LLVM AMDGPU</text>
<text class="tx dim" x="714" y="74" text-anchor="middle" lang="en">order · regs · waits</text><text class="tx dim" x="714" y="74" text-anchor="middle" lang="zh">顺序 · 寄存器 · wait</text>
<rect class="nd" x="804" y="32" width="80" height="56"/>
<text class="tx b" x="844" y="65" text-anchor="middle" lang="en">binary</text><text class="tx b" x="844" y="65" text-anchor="middle" lang="zh">二进制</text>
<path class="edge" d="M134,60 L146,60" marker-end="url(#ah2)"/><path class="edge" d="M266,60 L278,60" marker-end="url(#ah2)"/>
<path class="edge" d="M398,60 L410,60" marker-end="url(#ah2)"/><path class="edge" d="M530,60 L542,60" marker-end="url(#ah2)"/>
<path class="edge" d="M624,60 L636,60" marker-end="url(#ah2)"/><path class="edge" d="M790,60 L802,60" marker-end="url(#ah2)"/>
<text class="tx amber" x="148" y="108" lang="en">re-run on every call</text>
<text class="tx amber" x="148" y="108" lang="zh">每次调用都重跑</text>
<text class="tx red" x="280" y="124" lang="en">62K objects built, 9.9K kept</text>
<text class="tx red" x="280" y="124" lang="zh">建 62K 个对象，只留 9.9K</text>
<text class="tx amber" x="638" y="108" lang="en">{LLVM_SHARE} of cold compile</text>
<text class="tx amber" x="638" y="108" lang="zh">冷编译的 {LLVM_SHARE}</text>
<path class="edge red" d="M75,88 C75,175 714,175 714,90" marker-end="url(#ah2r)"/>
<text class="tx red" x="200" y="176" lang="en">the author overrides register use and order: fences, inline asm, pins, LLVM patches</text>
<text class="tx red" x="200" y="176" lang="zh">作者覆盖寄存器使用和指令顺序：fence、inline 汇编、pin、LLVM 补丁</text>
<text class="lane" x="16" y="206" lang="en">REDESIGN · ONE DECIDER PER DECISION</text>
<text class="lane" x="16" y="206" lang="zh">重新设计 · 每个决策一个决策者</text>
<text class="tx dim" x="382" y="214" lang="en">machine tables: AMD ISA files · latencies · hazards · atoms</text>
<text class="tx dim" x="382" y="214" lang="zh">机器描述表：AMD ISA 文件 · 延迟 · hazard · atom</text>
<rect class="nd teal" x="16" y="222" width="128" height="56"/><text class="tx b" x="80" y="246" text-anchor="middle">Python</text>
<text class="tx dim" x="80" y="264" text-anchor="middle" lang="en">generates only</text><text class="tx dim" x="80" y="264" text-anchor="middle" lang="zh">只做生成</text>
<rect class="nd teal" x="158" y="222" width="110" height="56"/><text class="tx b" x="213" y="246" text-anchor="middle">KIR</text>
<text class="tx dim" x="213" y="264" text-anchor="middle" lang="en">typed · tokens</text><text class="tx dim" x="213" y="264" text-anchor="middle" lang="zh">类型 · token</text>
<rect class="nd" x="282" y="222" width="86" height="56"/>
<text class="tx b" x="325" y="255" text-anchor="middle" lang="en">verify</text><text class="tx b" x="325" y="255" text-anchor="middle" lang="zh">检查</text>
<rect class="nd" x="382" y="222" width="86" height="56"/>
<text class="tx b" x="425" y="246" text-anchor="middle" lang="en">select</text><text class="tx b" x="425" y="246" text-anchor="middle" lang="zh">选指令</text>
<text class="tx dim" x="425" y="264" text-anchor="middle" lang="en">by table</text><text class="tx dim" x="425" y="264" text-anchor="middle" lang="zh">查表</text>
<rect class="nd" x="482" y="222" width="96" height="56"/>
<text class="tx b" x="530" y="246" text-anchor="middle" lang="en">order</text><text class="tx b" x="530" y="246" text-anchor="middle" lang="zh">排顺序</text>
<text class="tx dim" x="530" y="264" text-anchor="middle" lang="en">explicit|auto</text><text class="tx dim" x="530" y="264" text-anchor="middle" lang="zh">显式|自动</text>
<rect class="nd" x="592" y="222" width="96" height="56"/>
<text class="tx b" x="640" y="246" text-anchor="middle" lang="en">registers</text><text class="tx b" x="640" y="246" text-anchor="middle" lang="zh">分寄存器</text>
<text class="tx dim" x="640" y="264" text-anchor="middle" lang="en">pins kept</text><text class="tx dim" x="640" y="264" text-anchor="middle" lang="zh">保留 pin</text>
<rect class="nd" x="702" y="222" width="86" height="56"/>
<text class="tx b" x="745" y="246" text-anchor="middle" lang="en">waits</text><text class="tx b" x="745" y="246" text-anchor="middle" lang="zh">插 wait</text>
<text class="tx dim" x="745" y="264" text-anchor="middle" lang="en">from tokens</text><text class="tx dim" x="745" y="264" text-anchor="middle" lang="zh">由 token 推出</text>
<rect class="nd" x="802" y="222" width="82" height="56"/>
<text class="tx b" x="843" y="246" text-anchor="middle" lang="en">encode</text><text class="tx b" x="843" y="246" text-anchor="middle" lang="zh">编码</text>
<text class="tx dim" x="843" y="264" text-anchor="middle" lang="en">→ binary</text><text class="tx dim" x="843" y="264" text-anchor="middle" lang="zh">→ 二进制</text>
<path class="edge" d="M144,250 L156,250" marker-end="url(#ah2)"/><path class="edge" d="M268,250 L280,250" marker-end="url(#ah2)"/>
<path class="edge" d="M368,250 L380,250" marker-end="url(#ah2)"/><path class="edge" d="M468,250 L480,250" marker-end="url(#ah2)"/>
<path class="edge" d="M578,250 L590,250" marker-end="url(#ah2)"/><path class="edge" d="M688,250 L700,250" marker-end="url(#ah2)"/>
<path class="edge" d="M788,250 L800,250" marker-end="url(#ah2)"/>
<rect class="nd amber" x="282" y="318" width="506" height="46"/>
<text class="tx b" x="535" y="338" text-anchor="middle" lang="en">report: resources · pressure · waits · checks · settings</text>
<text class="tx b" x="535" y="338" text-anchor="middle" lang="zh">报告：资源 · 压力 · wait 来源 · 检查 · 设置</text>
<text class="tx dim" x="535" y="355" text-anchor="middle" lang="en">each step writes one section; no decision is hidden</text>
<text class="tx dim" x="535" y="355" text-anchor="middle" lang="zh">每个步骤写一节；没有隐藏的决定</text>
<path class="edge dash" d="M325,278 L325,316"/><path class="edge dash" d="M530,278 L530,316"/>
<path class="edge dash" d="M640,278 L640,316"/><path class="edge dash" d="M745,278 L745,316"/>
<rect class="nd teal" x="16" y="318" width="252" height="46"/>
<text class="tx b" x="142" y="338" text-anchor="middle" lang="en">interpret(): KIR on the CPU</text>
<text class="tx b" x="142" y="338" text-anchor="middle" lang="zh">interpret()：在 CPU 上跑 KIR</text>
<text class="tx dim" x="142" y="355" text-anchor="middle" lang="en">per-thread run · race checks</text>
<text class="tx dim" x="142" y="355" text-anchor="middle" lang="zh">逐线程执行 · 检查竞争</text>
<path class="edge dash" d="M213,278 L213,316"/>
</svg>
"""
