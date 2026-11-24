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

PLATE_LOOP = r"""
<svg viewBox="0 0 880 250" role="img" aria-labelledby="pl-t"><title id="pl-t">the loop: generate, change the code, teach the compiler</title>
<defs>
  <marker id="lh" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#14315c"/></marker>
</defs>
<text class="lane" x="16" y="20" lang="en">ONE KERNEL · COMPILER VERSION n</text>
<text class="lane" x="16" y="20" lang="zh">一个 KERNEL · 第 n 版编译器</text>
<rect class="nd" x="16" y="34" width="252" height="104"/>
<text class="tx b" x="30" y="60" lang="en">1 · Generate</text>
<text class="tx b" x="30" y="60" lang="zh">1 · 生成</text>
<text class="tx" x="30" y="84" lang="en">compile the kernel source</text>
<text class="tx" x="30" y="84" lang="zh">编译 kernel 源码</text>
<text class="tx dim" x="30" y="118">code(n) = f(llvm(n))</text>
<rect class="nd teal" x="314" y="34" width="252" height="104"/>
<text class="tx b" x="328" y="60" lang="en">2 · Change the code</text>
<text class="tx b" x="328" y="60" lang="zh">2 · 改代码</text>
<text class="tx" x="328" y="84" lang="en">edit the final code, measure</text>
<text class="tx" x="328" y="84" lang="zh">直接改最终代码，再测量</text>
<text class="tx dim" x="328" y="118">code(n+1) = agent(code(n))</text>
<rect class="nd amber" x="612" y="34" width="252" height="104"/>
<text class="tx b" x="626" y="60" lang="en">3 · Teach the compiler</text>
<text class="tx b" x="626" y="60" lang="zh">3 · 教会编译器</text>
<text class="tx" x="626" y="84" lang="en">it produces the same code</text>
<text class="tx" x="626" y="84" lang="zh">让它生成同样的代码</text>
<text class="tx dim" x="626" y="118">llvm(n+1) = f⁻¹(code(n+1))</text>
<path class="edge" d="M268,86 L312,86" marker-end="url(#lh)"/>
<path class="edge" d="M566,86 L610,86" marker-end="url(#lh)"/>
<text class="tx teal" x="589" y="162" text-anchor="middle" lang="en">only if correct and faster</text>
<text class="tx teal" x="589" y="162" text-anchor="middle" lang="zh">只在正确而且更快时</text>
<path class="edge" d="M738,138 L738,204 L142,204 L142,140" marker-end="url(#lh)"/>
<text class="tx" x="440" y="196" text-anchor="middle" lang="en">back to step 1: next kernel, compiler version n+1</text>
<text class="tx" x="440" y="196" text-anchor="middle" lang="zh">回到第 1 步：下一个 kernel，第 n+1 版编译器</text>
<text class="tx red" x="440" y="232" text-anchor="middle" lang="en">open question: this change must not slow other kernels (section 05)</text>
<text class="tx red" x="440" y="232" text-anchor="middle" lang="zh">未解决的问题：这个改动不能让其他 kernel 变慢（第 05 节）</text>
</svg>
"""
