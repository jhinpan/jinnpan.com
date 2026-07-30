import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Code,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  LineChart,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

/* ------------------------------------------------------------------ *
 * 下列数据要么是本次会话在本地 8x MI355X 节点上实测的，要么引自具名的
 * 公开来源。每一行都标注了出处，两类数据不混用。
 * ------------------------------------------------------------------ */

const MEASURED_RUNS: Array<{
  config: string;
  conc: number;
  totalTps: number;
  perGpu: number;
  ttftMs: number;
  tpotMs: number;
  peakBatch: string;
  wall: string;
}> = [
  {
    config: "非投机（官方配方）",
    conc: 96,
    totalTps: 6198.38,
    perGpu: 775,
    ttftMs: 29666,
    tpotMs: 91.0,
    peakBatch: "89",
    wall: "MLA KV 占用 0.99",
  },
  {
    config: "非投机（官方配方）",
    conc: 32,
    totalTps: 4919.04,
    perGpu: 615,
    ttftMs: 10743,
    tpotMs: 48.1,
    peakBatch: "32",
    wall: "未饱和",
  },
  {
    config: "DSPARK（官方配方）",
    conc: 48,
    totalTps: 2188.55,
    perGpu: 274,
    ttftMs: 10277,
    tpotMs: 168.7,
    peakBatch: "47",
    wall: "KDA 状态池 0.98",
  },
  {
    config: "DSPARK + ReplaySSM，mrr 128",
    conc: 48,
    totalTps: 2168.71,
    perGpu: 271,
    ttftMs: 2800,
    tpotMs: 174.6,
    peakBatch: "48",
    wall: "未饱和",
  },
  {
    config: "DSPARK（官方配方）",
    conc: 64,
    totalTps: 2176.25,
    perGpu: 272,
    ttftMs: 39748,
    tpotMs: 168.4,
    peakBatch: "47",
    wall: "KDA 状态池 + mrr=48",
  },
  {
    config: "DSPARK + HiCache（direct IO）",
    conc: 48,
    totalTps: 2126.41,
    perGpu: 266,
    ttftMs: 101059,
    tpotMs: 94.8,
    peakBatch: "26",
    wall: "KDA 状态池，4 份/请求",
  },
  {
    config: "DSPARK + ReplaySSM，mrr 128",
    conc: 96,
    totalTps: 1952.27,
    perGpu: 244,
    ttftMs: 4261,
    tpotMs: 372.1,
    peakBatch: "96",
    wall: "verify 开销 ∝ batch",
  },
];

/** 引自 sgl-project/sglang#32548，MI355X TP8，ISL 8192 / OSL 1024。 */
const PUBLISHED_AMD = {
  concurrencies: ["2", "4", "8", "16", "32"],
  nospec: [820.02, 1474.71, 2356.21, 3608.4, 4898.51],
  dspark: [1659.14, 2217.74, 2926.77, 3406.01, 3715.2],
};

const BLOCKERS: Array<{
  lever: string;
  whatItBuys: string;
  status: "blocked" | "works" | "caution";
  detail: string;
  cite: string;
}> = [
  {
    lever: "DCP（--dcp-size 8）",
    whatItBuys: "K3 上逻辑 MLA KV 提升 7.9 倍 —— 正是我们撞到的那道墙",
    status: "blocked",
    detail:
      "Kimi-K3 的 DCP 断言 decode 后端必须是 cutedsl_mla 或 tokenspeed_mla；两者都 gate 在 is_flashinfer_available() / is_blackwell 上，gfx950 一个都构造不出来。",
    cite: "arg_groups/overrides.py:397-399, utils/common.py:357,368",
  },
  {
    lever: "Verify-budget trimming",
    whatItBuys: "chat 流量下 bs 256 解码 +68%（LMSYS 声称）",
    status: "blocked",
    detail:
      "需要一个 advertise supports_ragged_verify_graph 的 full-attention 后端；hybrid 后端要求两半都支持，而 ROCm 侧没有任何 MLA 后端设了这个标志。此外折叠式 verify epilogue 另有 is_cuda() 门禁。",
    cite: "hybrid_linear_attn_backend.py:918-923, dspark_worker_v2.py:205-225",
  },
  {
    lever: "PP8 prefill lane",
    whatItBuys: "单节点 prefill 容量比 TEP8 高 1.45~1.72 倍",
    status: "blocked",
    detail:
      "DSPARK 硬性要求 pp_size == 1，所以 PP prefill lane 只能存在于不带 DSPARK 的部署里。",
    cite: "arg_groups/speculative_hook.py:303-306",
  },
  {
    lever: "HiCache（L1+L2 host tier）",
    whatItBuys: "把 prefix 复用扩展到显存 KV 之外",
    status: "caution",
    detail:
      "在 ROCm 上能启动、能服务，但必须加 --hicache-io-backend direct。默认的 kernel 后端会调用 transfer_kv_mamba_*，那两个核只在 is_cuda() 下 import —— 服务器会正常启动，然后在第一次 KDA offload 时挂掉。",
    cite: "mem_cache/memory_pool_host.py:31-45, server_args.py:7567-7571",
  },
  {
    lever: "ReplaySSM spec-verify",
    whatItBuys: "显存 KV +82%，KDA 池从 48 个槽升到约 100 个",
    status: "works",
    detail:
      "纯 Triton 的 ring 写入 + commit fold，没有 is_hip 门禁；nv_cutedsl verify 在 ROCm 上会静默映射到 Triton kernel。但没有任何 AMD CI 覆盖，精度需要本地验证。",
    cite: "kda_backend.py:97-98,870, mem_cache/memory_pool.py:659-675",
  },
  {
    lever: "--max-running-requests",
    whatItBuys: "解除排队上限（改善 TTFT，不改善吞吐）",
    status: "works",
    detail:
      "投机解码在该参数未显式设置时会静默把上限重置为 48。显式传入即可跳过重置。",
    cite: "arg_groups/speculative_hook.py:394-398",
  },
];

function Header() {
  const theme = useHostTheme();
  return (
    <Stack gap={6}>
      <H1>Kimi-K3 high-throughput：8x MI355X 实测 vs NVIDIA 公开数据</H1>
      <Text tone="secondary">
        单节点、TP8、ISL 8192 / OSL 1024、<Code>--dataset-name random</Code>
        （取自 ShareGPT 真实文本）。2026-07-28 在本地 ROCm 7.2 节点上针对{" "}
        <Code>sglang serve</Code> 实测；NVIDIA 侧数字引自 LMSYS 的 K3 day-0
        blog 与 SGLang K3 cookbook。
      </Text>
      <Row gap={6} wrap>
        <Pill size="sm">8x MI355X</Pill>
        <Pill size="sm">TP8 · bf16 · aiter MXFP4 MoE</Pill>
        <Pill size="sm">triton attention</Pill>
        <Pill size="sm" style={{ color: theme.text.tertiary }}>
          7 组 benchmark · 4 种服务器配置
        </Pill>
      </Row>
    </Stack>
  );
}

function Headline() {
  return (
    <Stack gap={12}>
      <Grid columns={4} gap={16}>
        <Stat value="775" label="tok/s/GPU — 我们实测的天花板" tone="success" />
        <Stat value="2,808" label="tok/s/GPU — NVIDIA 公开的最高点" />
        <Stat value="3.6x" label="聚合吞吐差距" tone="warning" />
        <Stat value="2.8x" label="高负载下 DSPARK 让我们损失的吞吐" tone="danger" />
      </Grid>
      <Callout tone="success" title="测量方法已经过验证">
        <Text>
          非投机配置在并发 32 实测 <Text weight="semibold">4,919 tok/s</Text>，
          而 sgl-project/sglang#32548 公布的是{" "}
          <Text weight="semibold">4,898 tok/s</Text> —— 相差 0.4%，中位 TPOT
          48.1 ms vs 48.73 ms。官方 AMD 数据完全可复现，所以下面 DSPARK
          那部分的差异是工作负载性质决定的，不是测量误差。
        </Text>
      </Callout>
    </Stack>
  );
}

function TheHeadlineFinding() {
  return (
    <Stack gap={10}>
      <H2>结论一：拖慢吞吐的是投机解码本身</H2>
      <Text tone="secondary">
        我们试过的每一种 DSPARK 配置都落在 1,952 ~ 2,189 tok/s
        之间。把投机关掉，同一台机器、同一个工作负载能跑到 6,198 tok/s。
        官方 AMD 的 sweep 其实已经显示了交叉点，但它停在并发 32 ——
        也就是两条曲线交叉之后的第一个点。
      </Text>
      <LineChart
        categories={PUBLISHED_AMD.concurrencies}
        series={[
          { name: "非投机（No-spec）", data: PUBLISHED_AMD.nospec, tone: "success" },
          { name: "DSpark", data: PUBLISHED_AMD.dspark, tone: "danger" },
        ]}
        valueSuffix=" tok/s"
        height={260}
        showValues
      />
      <Text size="small" tone="tertiary">
        总 token 吞吐 vs 请求并发数（横轴：并发请求数；纵轴：总 tokens/s，含
        input + output）。来源：sgl-project/sglang#32548，MI355X TP8，ISL 8192 /
        OSL 1024。DSpark 在并发 16 以下占优，之上转为劣势；并发 32 时比非投机慢
        24%。
      </Text>
      <Callout tone="info" title="交叉点为什么会出现，以及为什么对我们更不利">
        <Text>
          DSPARK 每步为每个请求验证 8 个 draft 位置，所以 target forward 的开销按{" "}
          <Code>batch x 8</Code> 增长，能不能划算全看 accept length。官方表里
          accept 是 5.29 ~ 5.93（低熵工作负载），而我们在 chat 类文本上实测只有{" "}
          <Text weight="semibold">2.5 ~ 3.3</Text>，这个数字恰好等于 LMSYS
          自己公布的 chat 流量数值（原文：accept length 约 2.7，所以典型一步里
          8 个验证位置有 5 个被浪费）。accept 2.5 意味着我们为 8 个位置付费只留下
          2.5 个 —— 这就是我们的 DSPARK 只到 2,189 tok/s，而官方 DSpark
          那一行写着 3,715 的原因。
        </Text>
      </Callout>
    </Stack>
  );
}

function TwoWalls() {
  const theme = useHostTheme();
  const wallStyle = {
    border: `1px solid ${theme.stroke.tertiary}`,
    borderRadius: 6,
    padding: 14,
  };
  return (
    <Stack gap={10}>
      <H2>结论二：两种配置撞的是两道不同的墙</H2>
      <Text tone="secondary">
        以下都是饱和时直接从 scheduler 日志读出来的。哪种资源先耗尽，决定了哪个
        上游手段才真正有用 —— 而非投机撞的那道墙，正好就是 NVIDIA 用 DCP
        抬高的那一道。
      </Text>
      <Grid columns={2} gap={16}>
        <Stack gap={8} style={wallStyle}>
          <H3>DSPARK 一侧 —— KDA 状态池</H3>
          <Text size="small" tone="secondary">
            并发 48 时：47 个在跑、1 个排队、
            <Text weight="semibold">mamba usage 0.98</Text>，而 MLA KV 只用到
            0.74。先耗尽的是每请求的 KDA 状态，不是 KV cache。
          </Text>
          <Divider />
          <Text size="small">
            ReplaySSM 能抬高这道墙：约 9 GB 的 <Code>intermediate_ssm</Code>{" "}
            verify 暂存被一个 2.45 GB 的 ring 取代，显存 KV 从 544,533 涨到
            990,344 token（+82%），KDA 池从约 48 个槽升到约 100 个。之后我们跑到
            96 个并发请求、零排队 —— 但吞吐并没有提升。
          </Text>
        </Stack>
        <Stack gap={8} style={wallStyle}>
          <H3>非投机一侧 —— MLA KV 容量</H3>
          <Text size="small" tone="secondary">
            并发 96 时：89 个在跑、7 个排队、
            <Text weight="semibold">MLA KV usage 0.99</Text>，而 mamba usage
            只有 0.24。该 batch 下 decode 稳定在 1,283 tok/s。
          </Text>
          <Divider />
          <Text size="small">
            833,536 个 KV token ÷ 每请求 9,216 token ={" "}
            <Text weight="semibold">约 90 个并发请求</Text>，实测正好停在这里。
            这就是 NVIDIA 用 DCP8 抬高 7.9 倍的那道墙（K3 上逻辑 KV 从 1.5M 到
            12.2M token）—— 而 DCP 对 Kimi-K3 在 ROCm 上是硬阻断的。
          </Text>
        </Stack>
      </Grid>
    </Stack>
  );
}

function HiCacheVerdict() {
  return (
    <Stack gap={10}>
      <H2>结论三：HiCache 在 ROCm 上能跑，但它不是吞吐手段</H2>
      <Text tone="secondary">
        它确实能开。我们去掉 <Code>--disable-radix-cache</Code>、加上{" "}
        <Code>--enable-hierarchical-cache --hicache-io-backend direct</Code>{" "}
        启动 Kimi-K3，日志显示{" "}
        <Code>impl=UnifiedRadixCache hybrid_ssm=True hierarchical=True</Code>，
        每 rank 分配 37.64 GB + 3.49 GB 主机池。它承接了真实流量，并且有 7.0%
        的缓存命中来自 host tier —— 说明 ROCm 上的 KDA host 传输在{" "}
        <Code>direct</Code> 后端下是通的。
      </Text>
      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader trailing={<Pill size="sm">无 prefix 复用</Pill>}>
            random 8192/1024，并发 48
          </CardHeader>
          <CardBody>
            <Grid columns={2} gap={12}>
              <Stat value="-2.9%" label="总吞吐相对基线" tone="danger" />
              <Stat value="9.8x" label="中位 TTFT 恶化倍数" tone="danger" />
            </Grid>
            <Divider style={{ marginTop: 12, marginBottom: 10 }} />
            <Text size="small" tone="secondary">
              2,126 vs 2,189 tok/s；中位 TTFT 101,059 ms vs 10,277 ms。原因是开
              radix cache 会把 KDA 策略切成 <Code>extra_buffer</Code>，每请求占 4
              份状态 —— 可运行 batch 从 47 掉到 26，另有 22 个请求排队。
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing={<Pill size="sm">4K 提示复用 8 次</Pill>}>
            generated-shared-prefix，并发 32
          </CardHeader>
          <CardBody>
            <Grid columns={2} gap={12}>
              <Stat value="45.9%" label="prompt token 缓存命中率" tone="success" />
              <Stat value="6,258" label="该工作负载下总 tok/s" />
            </Grid>
            <Divider style={{ marginTop: 12, marginBottom: 10 }} />
            <Text size="small" tone="secondary">
              1,107,048 个 prompt token 中有 507,968 来自缓存 —— 93% 来自
              device，7% 来自 HiCache 的 host tier。对照我们现在的配方，基准值是
              0%：<Code>--disable-radix-cache</Code> 意味着今天完全没有 prefix
              cache。
            </Text>
          </CardBody>
        </Card>
      </Grid>
      <Callout tone="warning" title="这里的一阶收益不是 HiCache，而是「有没有 prefix cache」">
        <Text>
          我们的官方配方带着 <Code>--disable-radix-cache</Code>。对无复用的
          benchmark 流量来说这个选择是对的，此时 HiCache 白扣 3% 吞吐；但对
          agentic 或多轮流量来说，它把每一个重复 prefix 都扔掉了。HiCache 是在
          radix cache 之上、当复用集超出显存 KV 时才产生价值的容量功能，不是加速
          功能。
        </Text>
      </Callout>
    </Stack>
  );
}

function GapToNvidia() {
  return (
    <Stack gap={10}>
      <H2>结论四：跟 B200/B300 差多少，以及差距背后到底是什么</H2>
      <BarChart
        categories={[
          "我们：非投机 c96（实测）",
          "官方 AMD：非投机 c32",
          "我们：DSPARK c48（实测）",
          "NVIDIA：PD-disagg 最高点",
        ]}
        series={[{ name: "单卡总吞吐", data: [775, 612, 274, 2808] }]}
        valueSuffix=" tok/s"
        height={240}
        showValues
      />
      <Text size="small" tone="tertiary">
        单卡总 token 吞吐，tokens/s/GPU（含 input + output）。「我们」与「官方
        AMD」两项均为 8x MI355X unified TP8、ISL 8192 / OSL 1024。NVIDIA
        那一根取自 LMSYS K3 blog 的 serving frontier：PD disaggregation 下一个
        PP8 prefill worker 喂一个 TP8 decode 节点、跑在 fp4 arm 上，ISL 与 OSL
        未披露。这不是同条件对比 —— 见下方说明。
      </Text>
      <Callout tone="warning" title="3.6 倍只能当作指示性数字，不能当同条件结论">
        <Text>
          目前不存在同一实验的 NVIDIA 公开数字。SGLang 的 K3 cookbook 里有
          B200/B300 配方，但每一条都标着 <Code>verified: false</Code>、验证仍在
          进行中，所以没有可对标的 B200/B300 实测 sweep。2,808 tok/s/GPU 是一个
          带独立 PP8 prefill lane 的 PD 分离组合，与我们的 unified TP8
          单节点在部署形态上就不同，而它依赖的两个并行手段我们都用不了。反过来也
          值得记一笔：cookbook 里 MI355X 和 MI350X 只有 Balanced 一档，压根没有
          High-Throughput 配方。
        </Text>
      </Callout>
    </Stack>
  );
}

function Levers() {
  return (
    <Stack gap={10}>
      <H2>NVIDIA 的吞吐手段里，我们究竟能用哪些</H2>
      <Table
        headers={["手段", "能带来什么", "ROCm 可用性", "原因 / 代码路径"]}
        rows={BLOCKERS.map((b) => [
          <Text weight="semibold">{b.lever}</Text>,
          <Text size="small">{b.whatItBuys}</Text>,
          <Pill size="sm" active={b.status === "works"}>
            {b.status === "blocked"
              ? "阻断"
              : b.status === "caution"
                ? "有条件"
                : "可用"}
          </Pill>,
          <Stack gap={4}>
            <Text size="small">{b.detail}</Text>
            <Text size="small" tone="quaternary">
              {b.cite}
            </Text>
          </Stack>,
        ])}
        rowTone={BLOCKERS.map((b) =>
          b.status === "blocked"
            ? "danger"
            : b.status === "caution"
              ? "warning"
              : "success",
        )}
        columnAlign={["left", "left", "center", "left"]}
      />
      <Text size="small" tone="tertiary">
        路径均相对于本地 checkout 的 <Code>python/sglang/srt/</Code>。目前没有任何
        registered AMD CI 覆盖 Kimi-K3、DSPARK、Kimi DCP 或 ReplaySSM —— 唯一的
        AMD DCP 测试是 Qwen3.5，而它不是 MLA 模型。
      </Text>
    </Stack>
  );
}

function AllRuns() {
  return (
    <Stack gap={10}>
      <H2>本次会话的全部实测数据</H2>
      <Table
        headers={[
          "服务器配置",
          "并发",
          "总 tok/s",
          "tok/s/GPU",
          "中位 TTFT (ms)",
          "中位 TPOT (ms)",
          "峰值 batch",
          "瓶颈所在",
        ]}
        rows={MEASURED_RUNS.map((r) => [
          r.config,
          r.conc,
          <Text weight="semibold">{r.totalTps.toLocaleString()}</Text>,
          r.perGpu,
          r.ttftMs.toLocaleString(),
          r.tpotMs.toFixed(1),
          r.peakBatch,
          <Text size="small" tone="secondary">
            {r.wall}
          </Text>,
        ])}
        columnAlign={[
          "left",
          "right",
          "right",
          "right",
          "right",
          "right",
          "right",
          "left",
        ]}
        striped
      />
      <Text size="small" tone="tertiary">
        所有实测统一条件：8x MI355X、TP8、ISL 8192 / OSL 1024、
        <Code>--dataset-name random --random-range-ratio 1</Code>、num-prompts =
        2 x 并发、<Code>--flush-cache</Code>。表中 mrr 指{" "}
        <Code>--max-running-requests</Code>。峰值 batch 与瓶颈来自 scheduler
        在饱和时的 decode-batch 日志行。
      </Text>
    </Stack>
  );
}

function Recommendations() {
  const items: Array<{ title: string; body: string; tag: string }> = [
    {
      tag: "收益最大",
      title: "把服务拆成两条 lane：吞吐走非投机，交互走 DSPARK",
      body:
        "同硬件同工作负载下 6,198 vs 2,189 tok/s。在这里 DSPARK 是一个 batch 1 到 batch 8 的功能 —— 官方表已经显示它在并发 16 以上转为劣势，而在 chat 熵的流量上我们 accept length 只有 2.5，情况更糟。不要用同一套配置同时承接两类流量。",
    },
    {
      tag: "数据披露",
      title: "把 #32548 的 sweep 延伸到并发 32 以上",
      body:
        "官方 sweep 在交叉点之后只多走了一个点，并且低估了我们非投机的天花板 26%（c32 的 4,898 对比实测 c96 的 6,198）。既然 high-throughput 才是关键指标，非投机那一栏应该补上 c64 与 c96。",
    },
    {
      tag: "建议采用（需验证）",
      title: "在 DSPARK lane 上打开 ReplaySSM",
      body:
        "--enable-linear-replayssm-spec 带来显存 KV +82%、KDA 池翻倍，并把并发 48 时的中位 TTFT 降低 73%（10,277 到 2,800 ms），吞吐持平。纯 Triton 实现、无 ROCm 门禁。但它没有 AMD CI 覆盖，采用前应先跑一轮本地 GSM8K/AIME 精度验证。",
    },
    {
      tag: "有条件",
      title: "只在有 prefix 复用的流量上开 HiCache，并强制 direct IO 后端",
      body:
        "在 ROCm 上默认的 kernel IO 后端会正常启动、然后在第一次 KDA offload 时失败，因为 transfer_kv_mamba_* 只在 is_cuda() 下 import。--hicache-io-backend direct 可用，其 host tier 承接了 7% 的命中。对无复用的 benchmark 流量则保留 --disable-radix-cache。",
    },
    {
      tag: "上游诉求",
      title: "真正的差距在于缺一条 ROCm 可用的 MLA DCP 路径",
      body:
        "MLA KV 在 89 个请求时就到 99%，而这正是 NVIDIA 用 DCP8 抬高 7.9 倍的部分，且 Kimi-K3 的 DCP 被焊死在 flashinfer/Blackwell 的 MLA decode kernel 上。要么提供一个 advertise supports_ragged_verify_graph 的 triton MLA 后端，要么为 ROCm 放宽 Kimi-K3 的 DCP 后端断言 —— 一次改动可同时解锁 DCP 和 verify trimming 两条路。",
    },
    {
      tag: "低成本实验",
      title: "缩小 DSPARK 的 draft window",
      body:
        "我们验证 8 个位置只留下 2.5 个。把 --speculative-dspark-block-size 降到 3（即 4 个 draft token）可按比例削减每步 verify 开销，而且跑在我们已经在用的 static verify 路径上。这是 ROCm 上唯一可用的 verify 降本手段，因为 SPS 表驱动的 trimming planner 在这里跑不起来。",
    },
  ];
  return (
    <Stack gap={10}>
      <H2>接下来该做什么</H2>
      {items.map((it) => (
        <div key={it.title}>
          <Card>
            <CardHeader trailing={<Pill size="sm">{it.tag}</Pill>}>
              {it.title}
            </CardHeader>
            <CardBody>
              <Text size="small">{it.body}</Text>
            </CardBody>
          </Card>
        </div>
      ))}
    </Stack>
  );
}

export default function KimiK3ThroughputReview() {
  return (
    <Stack gap={28} style={{ padding: 24, maxWidth: 1180 }}>
      <Header />
      <Headline />
      <Divider />
      <TheHeadlineFinding />
      <Divider />
      <TwoWalls />
      <Divider />
      <HiCacheVerdict />
      <Divider />
      <GapToNvidia />
      <Divider />
      <Levers />
      <Divider />
      <AllRuns />
      <Divider />
      <Recommendations />
    </Stack>
  );
}
