import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Code,
  CollapsibleSection,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  LineChart,
  Pill,
  Row,
  Select,
  Spacer,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasState,
} from "cursor/canvas";

type Lang = "zh" | "en";
type Bi = { zh: string; en: string };

function useT() {
  const [lang, setLang] = useCanvasState<Lang>("canvasLang", "zh");
  return { lang, setLang, t: (s: Bi) => s[lang] };
}

// ---------------------------------------------------------------------------
// Measured data. Kimi-K3 on 8x MI355X (gfx950), SGLang TP8, bf16 activations
// over MXFP4 experts, triton attention + triton linear-attn, radix cache off,
// batch size 1, no speculative decoding.
//
// Per-step numbers are device (GPU) time summed over the kernels of one decode
// forward pass on rank TP0, averaged over 24 consecutive decode steps.
// ---------------------------------------------------------------------------

const CTX = ["4K", "32K", "64K", "512K", "1M"];
const CTX_TOKENS = [4096, 32768, 65536, 524288, 1047552];

// ms per decode step, by cost centre
const KDA = [3.542, 3.552, 3.536, 3.593, 3.524];
const FULL_ATTN = [3.1, 4.053, 4.806, 12.671, 21.225];
const MOE = [10.548, 10.573, 10.547, 10.643, 10.521];
const ATTN_RES = [2.288, 2.291, 2.281, 2.328, 2.321];
const OTHER = [1.29, 1.325, 1.363, 1.924, 2.507];
const COMPUTE = [20.77, 21.79, 22.53, 31.16, 40.1];

// share of decode device time, %
const KDA_PCT = [17.1, 16.3, 15.7, 11.5, 8.8];
const FULL_PCT = [14.9, 18.6, 21.3, 40.7, 52.9];
const MOE_PCT = [50.8, 48.5, 46.8, 34.2, 26.2];
const ATTN_RES_PCT = [11.0, 10.5, 10.1, 7.5, 5.8];
const OTHER_PCT = [6.2, 6.1, 6.0, 6.2, 6.3];

// per-layer device time, microseconds
const KDA_PER_LAYER = [51.3, 51.4, 51.2, 52.1, 51.1];
const FULL_PER_LAYER = [129.2, 168.9, 200.3, 528.0, 884.4];
const PER_LAYER_RATIO = [2.5, 3.3, 3.9, 10.1, 17.3];

// measured end-to-end inter-token latency with CUDA graphs on (real config)
const REAL_ITL: (number | null)[] = [19.39, 20.39, 21.19, 29.67, 38.75];

// Index of the last context point with a measured latency, so nothing downstream
// asserts a number that was not measured.
const LAST_MEASURED = REAL_ITL.reduce<number>((acc, v, i) => (v === null ? acc : i), 0);

// MLA latent KV traffic and the bandwidth the KV-scan kernel achieves
const KV_GB = [0.11, 0.91, 1.81, 14.5, 28.96];
const KV_SCAN_MS = [0.308, 0.977, 1.532, 9.043, 17.609];
const KV_TBPS = [0.37, 0.93, 1.18, 1.6, 1.64];
const KV_PCT_PEAK = [4.6, 11.6, 14.8, 20.0, 20.6];
const HBM_PEAK_TBPS = 8.0;

// inside one KDA block, us per decode step across all 69 KDA layers (at 64K;
// identical at every context length within measurement noise)
const KDA_PARTS: { key: Bi; us: number }[] = [
  { key: { zh: "投影 GEMM (q/k/v/beta/gate/o)", en: "Projection GEMMs (q/k/v/beta/gate/o)" }, us: 2233 },
  { key: { zh: "递推状态更新 (KDA 核心)", en: "Recurrent state update (KDA core)" }, us: 445 },
  { key: { zh: "短卷积 (kernel_size=4)", en: "Short conv (kernel_size=4)" }, us: 297 },
  { key: { zh: "门控输出 RMSNorm", en: "Gated output RMSNorm" }, us: 269 },
  { key: { zh: "其他 (拷贝/逐元素)", en: "Other (copies / elementwise)" }, us: 292 },
];

// inside one full-attention (MLA) block, us per decode step across all 24 layers
const FA_PARTS: { key: Bi; at4k: number; at1m: number }[] = [
  { key: { zh: "KV 扫描 (stage 1)", en: "KV scan (stage 1)" }, at4k: 308, at1m: 17609 },
  { key: { zh: "KV-split 归约 (stage 2)", en: "KV-split reduction (stage 2)" }, at4k: 905, at1m: 1731 },
  { key: { zh: "投影 GEMM", en: "Projection GEMMs" }, at4k: 1029, at1m: 1036 },
  { key: { zh: "其他 (norm/门控/拼接)", en: "Other (norm / gate / concat)" }, at4k: 859, at1m: 849 },
];

// inside the MoE block, us per decode step across all 92 MoE layers
const MOE_PARTS: { key: Bi; us: number }[] = [
  { key: { zh: "共享专家 GEMM (稠密, 2 个)", en: "Shared-expert GEMMs (dense, 2 of them)" }, us: 3836 },
  { key: { zh: "路由 + 排序 + 量化", en: "Routing + sorting + quantization" }, us: 2959 },
  { key: { zh: "其他 (norm/归约/加法)", en: "Other (norm / reduce / add)" }, us: 2314 },
  { key: { zh: "被路由专家 GEMM (MXFP4, 16/896)", en: "Routed-expert GEMMs (MXFP4, 16 of 896)" }, us: 1439 },
];

const ATT_MIX: { key: Bi; pct: number }[] = [
  { key: { zh: "向量 ALU (VALU)", en: "Vector ALU (VALU)" }, pct: 55.7 },
  { key: { zh: "访存等待 (s_waitcnt)", en: "Memory wait (s_waitcnt)" }, pct: 23.5 },
  { key: { zh: "标量 / 控制流", en: "Scalar / control flow" }, pct: 11.9 },
  { key: { zh: "访存发射 (load/store)", en: "Memory issue (load/store)" }, pct: 5.0 },
  { key: { zh: "其他 / 分支", en: "Other / branch" }, pct: 4.0 },
];

const ATT_DIR =
  "/sgl-workspace/workspace/kda_prof/att_kda_64k/ui_output_agent_20495_dispatch_8";

// ---------------------------------------------------------------------------
// Prefill. Same server and same block ranges, but prefill never runs through a
// CUDA graph, so the profiler attributes its kernels directly. Collectives are
// broken out rather than dropped: a 16384-token chunk all-reduces 235 MB per
// layer, which is real work, unlike the 14 KB latency-bound all-reduce in decode.
// ---------------------------------------------------------------------------

const PF_CTX = ["1K", "4K", "8K", "32K"];

// ms of GPU time over the whole prefill, collectives excluded
const PF_KDA = [23.0, 58.5, 106.0, 414.4];
const PF_FULL = [6.1, 15.9, 34.9, 985.1];
const PF_MOE = [54.5, 125.3, 219.0, 791.6];
const PF_COLLECTIVE = [72.4, 77.3, 121.2, 429.1];
const PF_TOTAL_GPU = [169.6, 330.6, 588.5, 3052.0];

const PF_KDA_PCT = [23.6, 23.1, 22.7, 15.8];
const PF_FULL_PCT = [6.3, 6.3, 7.5, 37.6];
const PF_MOE_PCT = [56.1, 49.5, 46.9, 30.2];
const PF_OTHER_PCT = [14.0, 21.2, 23.0, 16.5];

// microseconds of GPU time per input token, collectives excluded
const PF_KDA_PER_TOK = [22.4, 14.3, 12.9, 12.6];
const PF_FULL_PER_TOK = [6.0, 3.9, 4.3, 30.1];
const PF_MOE_PER_TOK = [53.2, 30.6, 26.7, 24.2];
const PF_OTHER_PER_TOK = [13.3, 13.1, 13.1, 13.2];

// ms of GPU time attributable to one layer of each kind, over the whole prefill
const PF_KDA_PER_LAYER = [0.333, 0.848, 1.537, 6.005];
const PF_FULL_PER_LAYER = [0.255, 0.662, 1.454, 41.046];
const PF_LAYER_RATIO = [0.77, 0.78, 0.95, 6.83];

// the two kernels that carry each mechanism's prefill maths, ms
const PF_MLA_KERNEL = [1.2, 5.9, 18.5, 916.1];
const PF_KDA_CHUNK_KERNEL = [4.0, 19.5, 41.2, 176.9];

// ---------------------------------------------------------------------------
// The 32K prefill runs as two chunks of 16384. Splitting its trace at the
// forward-pass boundary shows every kernel identical between the two except the
// attention kernel, which costs 12.8x more in the second chunk at the same
// dispatch count. The recorded launch grid explains it: extend_attention_fwd
// launches (batch, heads, cdiv(max_extend_len, BLOCK_M)) and on gfx950 BLOCK_M
// is 128 only when 128 < Lq <= 256, else 64. Both chunks extend the same 16384
// tokens, so grid z of 128 vs 256 pins Lq at 192 vs 576 — the decompressed MHA
// form versus the absorbed latent form.
// ---------------------------------------------------------------------------
const CHUNK_SPLIT = {
  form: ["decompressed MHA", "absorbed latent"],
  lq: ["192 / 128", "576 / 512"],
  blockM: ["128", "64"],
  gridZ: ["[1, 12, 128]", "[1, 12, 256]"],
  pairs: [1.342e8, 4.027e8],
  flop: [1.031e12, 1.051e13],
  msPerLayer: [2.76, 35.41],
  totalMs: [66.3, 849.9],
  tflops: [373, 297],
};
// attention is 30.0% of GPU time in that trace but only 0.55% of its slices
const ATTN_SLICE_SHARE = 0.55;
const ATTN_TIME_SHARE = 30.0;

// ---------------------------------------------------------------------------
// Re-verification of the 32K point through Perfetto's trace_processor, i.e. a
// parser we did not write, plus a raw re-read of the chrome JSON. Three paths
// that share no code, so a bug in bucketize.py cannot reproduce itself.
// scripts/verify/ in the data archive runs all of them.
// ---------------------------------------------------------------------------
const VERIFY: { q: Bi; pub: string; raw: string; tp: string; band: string }[] = [
  { q: { zh: "GPU 派发数", en: "GPU dispatches" },
    pub: "8961", raw: "8961", tp: "8959", band: "—" },
  { q: { zh: "GPU 核函数总时间", en: "GPU kernel time" },
    pub: "3052.04 ms", raw: "3052.04 ms", tp: "3015.70 ms", band: "—" },
  { q: { zh: "_fwd_kernel (次数)", en: "_fwd_kernel (count)" },
    pub: "916.14 ms (48)", raw: "916.14 ms (48)", tp: "880.73 ms (47)", band: "—" },
  { q: { zh: "K3/full_attn 含集合通信", en: "K3/full_attn incl. collectives" },
    pub: "1031.97 ms", raw: "—", tp: "—", band: "1031.98 ms" },
  { q: { zh: "全注意力占计算时间", en: "full attn share of compute" },
    pub: "37.56%", raw: "37.56%", tp: "—", band: "37.57%" },
];

// Union of kernel intervals versus the naive sum: if they differ, summing
// durations double counts concurrent kernels. They do not.
const GPU_SUM_MS = 3052.04;
const GPU_UNION_MS = 3051.0;
const GPU_SPAN_MS = 3055.65;
const GPU_BUSY = 99.8;

// The same four ranges, measured on the host thread and on the device. The host
// blocks wherever the launch queue happens to fill, so CPU range width in prefill
// measures stalling, not cost -- and it points the opposite way.
const TRACK_INVERSION: { block: string; cpuMs: number; gpuMs: number }[] = [
  { block: "K3/kda", cpuMs: 1077.56, gpuMs: 546.09 },
  { block: "K3/moe", cpuMs: 154.2, gpuMs: 1035.54 },
  { block: "K3/full_attn", cpuMs: 31.13, gpuMs: 1031.97 },
  { block: "K3/dense_mlp", cpuMs: 0.82, gpuMs: 7.14 },
];
const CPU_TRACK_TOTAL = TRACK_INVERSION.reduce((a, r) => a + r.cpuMs, 0);
const GPU_TRACK_TOTAL = TRACK_INVERSION.reduce((a, r) => a + r.gpuMs, 0);

// Composition of each chunk on its own, GPU time inside the profiler's own
// gpu_user_annotation bands. Everything but attention is the same in both.
const CHUNK_COMPOSITION: { block: Bi; c1: number; c2: number }[] = [
  { block: { zh: "全注意力 (MLA)", en: "full attention (MLA)" }, c1: 120.6, c2: 911.82 },
  { block: { zh: "MoE FFN", en: "MoE FFN" }, c1: 526.51, c2: 524.12 },
  { block: { zh: "KDA", en: "KDA" }, c1: 276.75, c2: 272.07 },
];
const CHUNK_GPU_MS = [1135.36, 1916.59];
const CHUNK_WINDOW = ["0.00 – 1.14 s", "1.14 – 3.06 s"];
// share of each chunk's wall-clock window occupied by _fwd_kernel alone
const CHUNK_ATTN_DENSITY = [6.1, 45.4];

// Clean wall-clock reference, measured on a server built without the ranges
// (best of three, after a warm first call). Summed kernel time lands within 1% of
// it at every size, so prefill is GPU-saturated throughout and the composition
// below is the composition of wall-clock TTFT.
const PF_TTFT_MS = [170, 333, 593, 3051];
const PF_TOK_PER_S = [6041, 12299, 13814, 10741];
const PF_BUSY_OVER_TTFT = PF_TOTAL_GPU.map((g, i) => g / PF_TTFT_MS[i]);

function fmt(n: number, d = 2) {
  return n.toFixed(d);
}

// ---------------------------------------------------------------------------

function Header() {
  const { lang, setLang, t } = useT();
  return (
    <Stack gap={10}>
      <Row align="center" gap={12}>
        <H1>
          {t({
            zh: "Kimi K3 的时间去哪了：KDA vs 全注意力 vs MoE（decode 与 prefill）",
            en: "Where Kimi K3 spends its time: KDA vs full attention vs MoE, in decode and prefill",
          })}
        </H1>
        <Spacer />
        <Select
          value={lang}
          onChange={(v) => setLang(v as Lang)}
          options={[
            { value: "zh", label: "中文" },
            { value: "en", label: "English" },
          ]}
        />
      </Row>
      <Row gap={6} wrap>
        <Pill size="sm">8 × AMD Instinct MI355X (gfx950)</Pill>
        <Pill size="sm">SGLang TP8</Pill>
        <Pill size="sm">bf16 act / MXFP4 experts</Pill>
        <Pill size="sm">{t({ zh: "批大小 1", en: "batch size 1" })}</Pill>
        <Pill size="sm">{t({ zh: "无投机解码", en: "no speculative decoding" })}</Pill>
        <Pill size="sm">{t({ zh: "93 层 = 69 KDA + 24 全注意力", en: "93 layers = 69 KDA + 24 full-attn" })}</Pill>
      </Row>
      {lang === "zh" ? (
        <Text tone="secondary">
          你观察到的 3:1 是对的：93 层里 69 层是 KDA，24 层是全注意力（MLA），
          而且每 4 层里前 3 层都是 KDA。但<b>层数占比不等于时间占比</b>。KDA 的状态大小与上下文无关，
          MLA 的 KV cache 随上下文线性增长，所以两者的时间占比会随上下文剧烈分化：
          KDA 占解码时间从 4K 的 <Code>17.1%</Code> 一路降到 1M 的 <Code>8.8%</Code>，
          而全注意力从 <Code>14.9%</Code> 涨到 <Code>52.9%</Code>。
          页面先讲 decode（4K–1M），再讲 prefill（1K–32K）——两个阶段的答案方向一致，但交叉点不同。
        </Text>
      ) : (
        <Text tone="secondary">
          Your 3:1 observation holds: 69 of the 93 layers are KDA and 24 are full attention
          (MLA), with the first three of every four layers being KDA. But{" "}
          <b>a share of layers is not a share of time</b>. KDA's state size is independent of
          context while the MLA KV cache grows linearly with it, so the two diverge sharply:
          KDA falls from <Code>17.1%</Code> of decode time at 4K to <Code>8.8%</Code> at 1M,
          while full attention climbs from <Code>14.9%</Code> to <Code>52.9%</Code>. Decode
          (4K–1M) comes first below, then prefill (1K–32K) — the two phases point the same way
          but cross over at different lengths.
        </Text>
      )}
    </Stack>
  );
}

function Hero() {
  const { t } = useT();
  return (
    <Grid columns={4} gap={16}>
      <Stat
        value="17.1% → 8.8%"
        label={t({ zh: "KDA 占解码时间 (4K → 1M)", en: "KDA share of decode time (4K → 1M)" })}
        tone="info"
      />
      <Stat
        value="3.54 → 3.52 ms"
        label={t({ zh: "69 层 KDA 绝对耗时，几乎不变", en: "69 KDA layers, absolute cost — flat" })}
        tone="success"
      />
      <Stat
        value="6.8×"
        label={t({ zh: "24 层全注意力耗时增长 (4K → 1M)", en: "24 full-attn layers, growth (4K → 1M)" })}
        tone="danger"
      />
      <Stat
        value={`${fmt(REAL_ITL[0] as number, 1)} → ${fmt(REAL_ITL[LAST_MEASURED] as number, 1)} ms`}
        label={t({
          zh: `实测每步解码延迟 (4K → ${CTX[LAST_MEASURED]})`,
          en: `Measured decode latency per step (4K → ${CTX[LAST_MEASURED]})`,
        })}
      />
    </Grid>
  );
}

function MainChart() {
  const { t } = useT();
  const series = [
    { name: t({ zh: "KDA (69 层)", en: "KDA (69 layers)" }), data: KDA, tone: "info" as const },
    { name: t({ zh: "全注意力 MLA (24 层)", en: "Full attention MLA (24 layers)" }), data: FULL_ATTN, tone: "danger" as const },
    { name: t({ zh: "MoE FFN (92 层)", en: "MoE FFN (92 layers)" }), data: MOE, tone: "warning" as const },
    { name: t({ zh: "注意力残差库", en: "Attention-residual bank" }), data: ATTN_RES, tone: "success" as const },
    { name: t({ zh: "其他 (norm/embed/采样)", en: "Other (norm / embed / sampling)" }), data: OTHER, tone: "neutral" as const },
  ];
  return (
    <Stack gap={8}>
      <H2>{t({ zh: "每步解码的设备时间构成", en: "Device time per decode step" })}</H2>
      <Text size="small" tone="tertiary">
        {t({
          zh: "纵轴：每步解码 GPU 时间 (ms) · 横轴：上下文长度 (tokens) · 堆叠为按块类型归类的核函数时间之和",
          en: "y: GPU time per decode step (ms) · x: context length (tokens) · stacked sum of kernel time grouped by block type",
        })}
      </Text>
      <BarChart
        categories={CTX}
        series={series}
        stacked
        height={300}
        valueSuffix=" ms"
      />
      <Text size="small" tone="tertiary">
        {t({
          zh: "来源：SGLang torch profiler 内核轨迹，rank TP0，24 个连续解码步平均，2026-07-30。MoE 与 KDA 是水平线；只有全注意力在增长。",
          en: "Source: SGLang torch-profiler kernel trace, rank TP0, mean of 24 consecutive decode steps, 2026-07-30. MoE and KDA are flat lines; only full attention grows.",
        })}
      </Text>
      <Callout
        tone="info"
        title={t({ zh: "读法", en: "How to read this" })}
      >
        {t({
          zh: "KDA 和 MoE 的条段高度在 5 个上下文点上几乎完全相同——它们都不随上下文增长。整根柱子变高，完全是全注意力那一段撑起来的。",
          en: "The KDA and MoE bands are essentially the same height at all five context points — neither grows with context. The entire increase in total height comes from the full-attention band.",
        })}
      </Callout>
    </Stack>
  );
}

function ShareAndPerLayer() {
  const { t } = useT();
  return (
    <Grid columns={2} gap={24}>
      <Stack gap={8}>
        <H3>{t({ zh: "占比：谁在吃解码时间", en: "Share: who eats decode time" })}</H3>
        <Text size="small" tone="tertiary">
          {t({
            zh: "纵轴：占每步解码设备时间的百分比 (%) · 横轴：上下文长度 (tokens)",
            en: "y: percent of device time per decode step (%) · x: context length (tokens)",
          })}
        </Text>
        <BarChart
          categories={CTX}
          series={[
            { name: t({ zh: "KDA", en: "KDA" }), data: KDA_PCT, tone: "info" },
            { name: t({ zh: "全注意力 MLA", en: "Full attention MLA" }), data: FULL_PCT, tone: "danger" },
            { name: t({ zh: "MoE FFN", en: "MoE FFN" }), data: MOE_PCT, tone: "warning" },
            { name: t({ zh: "注意力残差库", en: "Attention-residual bank" }), data: ATTN_RES_PCT, tone: "success" },
            { name: t({ zh: "其他", en: "Other" }), data: OTHER_PCT, tone: "neutral" },
          ]}
          stacked
          height={260}
          valueSuffix="%"
        />
      </Stack>
      <Stack gap={8}>
        <H3>{t({ zh: "单层成本：一层 KDA vs 一层全注意力", en: "Per-layer cost: one KDA layer vs one full-attn layer" })}</H3>
        <Text size="small" tone="tertiary">
          {t({
            zh: "纵轴：单层每步设备时间 (μs，对数感知的线性轴) · 横轴：上下文长度 (tokens)",
            en: "y: device time per layer per step (μs) · x: context length (tokens)",
          })}
        </Text>
        <LineChart
          categories={CTX}
          series={[
            { name: t({ zh: "一层 KDA", en: "One KDA layer" }), data: KDA_PER_LAYER, tone: "info" },
            { name: t({ zh: "一层全注意力 (MLA)", en: "One full-attn (MLA) layer" }), data: FULL_PER_LAYER, tone: "danger" },
          ]}
          height={260}
          valueSuffix=" μs"
          showValues
        />
        <Text size="small" tone="tertiary">
          {t({
            zh: "同样是一层，全注意力在 4K 时已是 KDA 的 2.5 倍，到 1M 是 17.3 倍。",
            en: "Layer for layer, full attention already costs 2.5× a KDA layer at 4K, and 17.3× at 1M.",
          })}
        </Text>
      </Stack>
    </Grid>
  );
}

function NumbersTable() {
  const { t } = useT();
  const rows = CTX.map((c, i) => [
    <Text weight="semibold">{c}</Text>,
    CTX_TOKENS[i].toLocaleString(),
    `${fmt(KDA[i])} (${fmt(KDA_PCT[i], 1)}%)`,
    `${fmt(FULL_ATTN[i])} (${fmt(FULL_PCT[i], 1)}%)`,
    `${fmt(MOE[i])} (${fmt(MOE_PCT[i], 1)}%)`,
    `${fmt(ATTN_RES[i])} (${fmt(ATTN_RES_PCT[i], 1)}%)`,
    `${fmt(OTHER[i])} (${fmt(OTHER_PCT[i], 1)}%)`,
    fmt(COMPUTE[i]),
    REAL_ITL[i] === null ? "—" : fmt(REAL_ITL[i] as number),
  ]);
  return (
    <Stack gap={8}>
      <H2>{t({ zh: "完整数值", en: "Full numbers" })}</H2>
      <Text size="small" tone="tertiary">
        {t({
          zh: "每步解码的 GPU 时间 (ms)，括号内为占设备时间的百分比。最后一列是 CUDA graph 打开时的实测 token 间延迟。",
          en: "GPU time per decode step (ms); the percentage in brackets is the share of device time. The last column is measured inter-token latency with CUDA graphs enabled.",
        })}
      </Text>
      <Table
        headers={[
          t({ zh: "上下文", en: "Context" }),
          t({ zh: "tokens", en: "Tokens" }),
          t({ zh: "KDA (69 层)", en: "KDA (69 L)" }),
          t({ zh: "全注意力 (24 层)", en: "Full attn (24 L)" }),
          t({ zh: "MoE (92 层)", en: "MoE (92 L)" }),
          t({ zh: "注意力残差", en: "Attn residual" }),
          t({ zh: "其他", en: "Other" }),
          t({ zh: "设备时间合计", en: "Device total" }),
          t({ zh: "实测延迟/步", en: "Measured ms/step" }),
        ]}
        rows={rows}
        columnAlign={["left", "right", "right", "right", "right", "right", "right", "right", "right"]}
        striped
      />
      <Text size="small" tone="tertiary">
        {t({
          zh: "设备时间合计与实测延迟之比稳定在 0.93–0.94，说明按核函数归类得到的构成可以直接当作真实解码时间的构成来读。",
          en: "Measured latency divided by device-time total is a steady 0.93–0.94, so the kernel-level composition can be read directly as the composition of real decode time.",
        })}
      </Text>
    </Stack>
  );
}

function WhyDiverge() {
  const { lang, t } = useT();
  return (
    <Stack gap={12}>
      <H2>{t({ zh: "为什么会分化：一个是常量，一个是线性", en: "Why they diverge: one is constant, the other linear" })}</H2>
      <Grid columns={2} gap={24}>
        <Stack gap={10}>
          {lang === "zh" ? (
            <Text>
              KDA 每层每卡只保存一个 <Code>[128 × 128]</Code> 的递推状态（每卡 12 个头），
              69 层加起来是 <b>54.3 MB</b>，而且<b>不随上下文变化</b>——解码一步就是把这块状态读进来、
              做一次带 per-K 衰减的 delta-rule 更新、再写回去。
              MLA 则要保存压缩后的 latent KV：每 token 每卡 <Code>24 × 576 × 2B = 27.0 KiB</Code>，
              到 1M 上下文时一步解码要读 <b>28.96 GB</b>。两者相差 500 倍以上。
            </Text>
          ) : (
            <Text>
              KDA keeps one <Code>[128 × 128]</Code> recurrent state per layer per rank (12 heads
              per rank). Across 69 layers that is <b>54.3 MB</b>, and it{" "}
              <b>does not change with context</b> — a decode step reads that state, applies one
              delta-rule update with per-K decay, and writes it back. MLA instead keeps a
              compressed latent KV cache of <Code>24 × 576 × 2B = 27.0 KiB</Code> per token per
              rank, so at 1M context a single decode step must read <b>28.96 GB</b>. That is a
              gap of more than 500×.
            </Text>
          )}
          <Callout tone="warning" title={t({ zh: "但 6.8 倍 ≠ 500 倍", en: "But 6.8× is not 500×" })}>
            {t({
              zh: "读的字节多 500 倍，耗时只多 6.8 倍——因为 KDA 的核函数完全受延迟限制（只有 48 个 workgroup，用不满 GPU），而 MLA 的 KV 扫描是受带宽限制的，能把机器跑起来。两者处在完全不同的瓶颈区间。",
              en: "500× more bytes but only 6.8× more time, because the KDA kernel is entirely latency-bound (only 48 workgroups — it cannot fill the GPU) while the MLA KV scan is bandwidth-bound and does put the machine to work. The two sit in completely different bottleneck regimes.",
            })}
          </Callout>
        </Stack>
        <Stack gap={8}>
          <H3>{t({ zh: "MLA KV 扫描达到的带宽", en: "Bandwidth achieved by the MLA KV scan" })}</H3>
          <Text size="small" tone="tertiary">
            {t({
              zh: "纵轴：达到的 HBM 读带宽占峰值百分比 (%)，峰值按 MI355X 8 TB/s 计 · 横轴：上下文长度 (tokens)",
              en: "y: achieved HBM read bandwidth as percent of peak (%), peak taken as MI355X 8 TB/s · x: context length (tokens)",
            })}
          </Text>
          <BarChart
            categories={CTX}
            series={[
              {
                name: t({ zh: "KV 扫描达到的带宽占峰值", en: "KV scan, % of peak bandwidth" }),
                data: KV_PCT_PEAK,
                tone: "danger",
              },
            ]}
            height={200}
            valueSuffix="%"
            yMax={100}
            referenceLines={[
              { value: 100, label: t({ zh: "8 TB/s 峰值", en: "8 TB/s peak" }), tone: "neutral" },
            ]}
          />
          <Text size="small" tone="tertiary">
            {t({
              zh: "即使在最有利的 1M 点上也只达到 1.64 TB/s（20.6%）。这是本次测量里最大的一块可优化空间。",
              en: "Even at the most favourable point (1M) it reaches only 1.64 TB/s, i.e. 20.6%. This is the largest single optimization headroom in the measurement.",
            })}
          </Text>
        </Stack>
      </Grid>
      <Table
        headers={[
          t({ zh: "上下文", en: "Context" }),
          t({ zh: "每步读取的 latent KV", en: "Latent KV read per step" }),
          t({ zh: "KV 扫描核函数耗时", en: "KV-scan kernel time" }),
          t({ zh: "达到的带宽", en: "Achieved bandwidth" }),
          t({ zh: "占 8 TB/s 峰值", en: "% of 8 TB/s peak" }),
        ]}
        rows={CTX.map((c, i) => [
          <Text weight="semibold">{c}</Text>,
          `${fmt(KV_GB[i])} GB`,
          `${fmt(KV_SCAN_MS[i], 3)} ms`,
          `${fmt(KV_TBPS[i])} TB/s`,
          `${fmt(KV_PCT_PEAK[i], 1)}%`,
        ])}
        columnAlign={["left", "right", "right", "right", "right"]}
        striped
      />
    </Stack>
  );
}

function InsideKDA() {
  const { t } = useT();
  const total = KDA_PARTS.reduce((s, p) => s + p.us, 0);
  return (
    <Card>
      <CardHeader trailing={<Pill size="sm">{`${fmt(total / 1000)} ms / step`}</Pill>}>
        {t({ zh: "拆开一层 KDA：真正的递推只占 12.6%", en: "Inside a KDA layer: the recurrence itself is only 12.6%" })}
      </CardHeader>
      <CardBody>
        <Stack gap={12}>
          <Text size="small" tone="tertiary">
            {t({
              zh: "横轴：全部 69 层 KDA 每步的设备时间 (μs) · 纵轴：KDA 块内部的组成部分 · 数据取自 64K，其余上下文点在测量噪声内相同",
              en: "x: device time per step across all 69 KDA layers (μs) · y: components inside the KDA block · measured at 64K; identical at the other context points within noise",
            })}
          </Text>
          <BarChart
            categories={KDA_PARTS.map((p) => t(p.key))}
            series={[
              {
                name: t({ zh: "每步设备时间", en: "Device time per step" }),
                data: KDA_PARTS.map((p) => p.us),
              },
            ]}
            horizontal
            height={220}
            valueSuffix=" μs"
          />
          <Callout tone="info" title={t({ zh: "值得注意", en: "Worth noting" })}>
            {t({
              zh: "如果目标是压 KDA 的开销，递推核函数不是靶子——它只占 KDA 时间的 12.6%（每层 6.5 μs）。63% 都在投影 GEMM 上，那是普通的 bf16 矩阵乘，和线性注意力本身无关。",
              en: "If the goal is to cut KDA overhead, the recurrence kernel is not the target — it is 12.6% of KDA time (6.5 μs per layer). 63% sits in the projection GEMMs, which are ordinary bf16 matmuls and have nothing to do with linear attention as such.",
            })}
          </Callout>
        </Stack>
      </CardBody>
    </Card>
  );
}

function InsideFullAttn() {
  const { t } = useT();
  return (
    <Stack gap={8}>
      <H3>{t({ zh: "拆开一层全注意力：瓶颈随上下文换位", en: "Inside a full-attention layer: the bottleneck moves with context" })}</H3>
      <Text size="small" tone="tertiary">
        {t({
          zh: "纵轴：全部 24 层全注意力每步的设备时间 (μs) · 横轴：全注意力块内部的组成部分 · 两个系列分别是 4K 和 1M 上下文",
          en: "y: device time per step across all 24 full-attention layers (μs) · x: components inside the full-attention block · the two series are 4K and 1M context",
        })}
      </Text>
      <BarChart
        categories={FA_PARTS.map((p) => t(p.key))}
        series={[
          { name: t({ zh: "4K 上下文", en: "4K context" }), data: FA_PARTS.map((p) => p.at4k), tone: "info" },
          { name: t({ zh: "1M 上下文", en: "1M context" }), data: FA_PARTS.map((p) => p.at1m), tone: "danger" },
        ]}
        height={240}
        valueSuffix=" μs"
      />
      <Text size="small" tone="tertiary">
        {t({
          zh: "在 4K，KV 扫描只占全注意力时间的 9.9%，投影和 KV-split 归约才是主项；到 1M，KV 扫描占 83%，其余三项几乎一字不变。这也解释了为什么 4K→64K 增长很温和，而 64K 之后开始陡升。",
          en: "At 4K the KV scan is just 9.9% of full-attention time — projections and the KV-split reduction dominate. At 1M the KV scan is 83% while the other three components are essentially unchanged. That is why growth is mild from 4K to 64K and steep afterwards.",
        })}
      </Text>
    </Stack>
  );
}

function InsideMoE() {
  const { t } = useT();
  return (
    <Stack gap={8}>
      <H3>{t({ zh: "顺带一个意外：batch=1 时 MoE 的钱没花在专家上", en: "An aside: at batch 1, MoE time is not spent on the experts" })}</H3>
      <Table
        headers={[
          t({ zh: "MoE 块内部", en: "Inside the MoE block" }),
          t({ zh: "每步设备时间", en: "Device time per step" }),
          t({ zh: "占 MoE", en: "Share of MoE" }),
        ]}
        rows={MOE_PARTS.map((p) => [
          t(p.key),
          `${fmt(p.us / 1000, 3)} ms`,
          `${fmt((100 * p.us) / MOE_PARTS.reduce((s, q) => s + q.us, 0), 1)}%`,
        ])}
        columnAlign={["left", "right", "right"]}
        striped
      />
      <Text size="small" tone="tertiary">
        {t({
          zh: "896 个专家里每 token 只激活 16 个，所以被路由专家的 MXFP4 GEMM 只有 1.44 ms（13.6%）；两个稠密共享专家反而花了 3.84 ms，路由与排序又花了 2.96 ms。MoE 在所有上下文点都是 10.5 ms 的常量项。",
          en: "Only 16 of 896 experts are active per token, so the routed-expert MXFP4 GEMMs cost just 1.44 ms (13.6%). The two dense shared experts cost 3.84 ms and routing plus sorting another 2.96 ms. MoE is a constant 10.5 ms term at every context point.",
        })}
      </Text>
    </Stack>
  );
}

function AttTrace() {
  const { lang, t } = useT();
  return (
    <Stack gap={12}>
      <H2>{t({ zh: "KDA 的 RCV / ATT 指令级轨迹（64K 配置）", en: "RCV / ATT instruction-level trace of KDA (64K configuration)" })}</H2>
      {lang === "zh" ? (
        <Text tone="secondary">
          用 <Code>rocprofv3 --att</Code> 抓了 KDA 解码核函数
          <Code>fused_recurrent_kda_packed_decode_kernel</Code> 的 Advanced Thread Trace，
          已由 ROCprof Trace Decoder 解码成 ROCprof Compute Viewer (RCV) 可直接打开的目录。
          因为这个核函数的启动形状与上下文无关，64K 的轨迹与任何其他上下文长度下的轨迹是同一个。
        </Text>
      ) : (
        <Text tone="secondary">
          An Advanced Thread Trace of the KDA decode kernel{" "}
          <Code>fused_recurrent_kda_packed_decode_kernel</Code> was captured with{" "}
          <Code>rocprofv3 --att</Code> and decoded by the ROCprof Trace Decoder into a directory
          that ROCprof Compute Viewer (RCV) opens directly. Because the kernel's launch geometry
          is independent of context, the 64K trace is the same trace as at any other context
          length.
        </Text>
      )}
      <Grid columns={4} gap={16}>
        <Stat value="48" label={t({ zh: "workgroup 数 (4 × 12 头)", en: "workgroups (4 × 12 heads)" })} tone="warning" />
        <Stat value="19%" label={t({ zh: "被占用的 CU 比例 (48 / 256)", en: "CUs occupied (48 of 256)" })} tone="warning" />
        <Stat value="4.91 μs" label={t({ zh: "独立测得的核函数设备时间", en: "standalone kernel device time" })} />
        <Stat value="6.48 μs" label={t({ zh: "服务器内实测每层耗时", en: "in-situ device time per layer" })} />
      </Grid>
      <Grid columns={2} gap={24}>
        <Stack gap={8}>
          <H3>{t({ zh: "波内指令时间构成", en: "Instruction time mix within a wave" })}</H3>
          <Text size="small" tone="tertiary">
            {t({
              zh: "纵轴：占被追踪波的总延迟周期百分比 (%) · 横轴：指令类别 · 共 1439 个指令槽、61216 个周期",
              en: "y: percent of total latency cycles in the traced wave (%) · x: instruction class · 1439 instruction slots, 61216 cycles",
            })}
          </Text>
          <BarChart
            categories={ATT_MIX.map((m) => t(m.key))}
            series={[
              { name: t({ zh: "占延迟周期", en: "% of latency cycles" }), data: ATT_MIX.map((m) => m.pct) },
            ]}
            height={220}
            valueSuffix="%"
          />
        </Stack>
        <Stack gap={10}>
          <H3>{t({ zh: "轨迹告诉我们什么", en: "What the trace says" })}</H3>
          {lang === "zh" ? (
            <Text>
              单个波内部 <b>55.7%</b> 的时间是向量 ALU——对每个头的 <Code>[128×128]</Code> fp32
              状态做衰减和 delta-rule 更新是实打实的算术；<b>23.5%</b> 是
              <Code>s_waitcnt</Code> 访存等待，全部计为 stall，整体 stall 率 30.1%。
              也就是说：<b>每个波都很忙，但波太少</b>。48 个 workgroup、每个 64 线程，
              在 256 个 CU 的 MI355X 上只用到 19% 的 CU、每个 CU 一个波，
              于是这个核函数不管怎么优化单波效率，都还是 5–6 μs 起步。
            </Text>
          ) : (
            <Text>
              Inside a single wave, <b>55.7%</b> of the time is vector ALU — decaying and
              delta-rule-updating a <Code>[128×128]</Code> fp32 state per head is real
              arithmetic. <b>23.5%</b> is <Code>s_waitcnt</Code> memory wait, all of it counted
              as stall, for an overall stall rate of 30.1%. So{" "}
              <b>each wave is busy, but there are far too few waves</b>: 48 workgroups of 64
              threads occupy 19% of the 256 CUs on an MI355X with one wave each. However well
              the per-wave efficiency is tuned, the kernel still costs 5–6 μs.
            </Text>
          )}
          <Callout tone="neutral" title={t({ zh: "轨迹文件", en: "Trace files" })}>
            <Stack gap={4}>
              <Text size="small">
                {t({ zh: "RCV 可直接打开的目录：", en: "Directory RCV opens directly:" })}
              </Text>
              <Code>{ATT_DIR}</Code>
              <Text size="small" tone="tertiary">
                {t({
                  zh: "内含 8 个 shader engine 的逐波 JSON、occupancy.json、wstates*.json、code.json 与 Triton 源码对应文件；同目录还有原始 .att / .out 与 rocpd SQLite 库。",
                  en: "Contains per-wave JSON for all 8 shader engines, occupancy.json, wstates*.json, code.json and the Triton source-correlation files; the parent directory also holds the raw .att / .out files and a rocpd SQLite database.",
                })}
              </Text>
            </Stack>
          </Callout>
        </Stack>
      </Grid>
    </Stack>
  );
}

function PrefillSection() {
  const { lang, t } = useT();
  return (
    <Stack gap={14}>
      <H2>{t({ zh: "换到 Prefill：同样的 3:1，答案几乎反过来", en: "Switching to prefill: same 3:1 layers, nearly the opposite answer" })}</H2>
      {lang === "zh" ? (
        <Text tone="secondary">
          Prefill 不走 CUDA graph，所以 profiler 能直接看到每个核函数，块区间可以直接归因，
          不需要解码那套名字映射。而且求和得到的核函数时间在四个点上都落在实测 TTFT 的 1% 以内
          （169.6/170、330.6/333、588.5/593、3052/3051），所以下面的构成就是 TTFT 的构成，
          集合通信也一并计入——prefill 的 all-reduce 每层要搬 235 MB（16384 token 的 chunk），是真实开销，
          不像解码时那 14 KB 的延迟型 all-reduce。
        </Text>
      ) : (
        <Text tone="secondary">
          Prefill never goes through a CUDA graph, so the profiler sees every kernel and the
          block ranges attribute them directly — none of the name-map machinery decode needed.
          Summed kernel time also lands within 1% of measured TTFT at all four points
          (169.6/170, 330.6/333, 588.5/593, 3052/3051), so what follows is the composition of
          wall-clock TTFT, collectives included. Prefill all-reduces are real work: a
          16384-token chunk moves 235 MB per layer, unlike the 14 KB latency-bound all-reduce
          in decode.
        </Text>
      )}
      <Grid columns={4} gap={16}>
        <Stat
          value="23.6% → 15.8%"
          label={t({ zh: "KDA 占 prefill 时间 (1K → 32K)", en: "KDA share of prefill time (1K → 32K)" })}
          tone="info"
        />
        <Stat
          value="0.77× → 6.8×"
          label={t({ zh: "单层成本比 全注意力:KDA", en: "Per-layer cost ratio, full-attn : KDA" })}
          tone="danger"
        />
        <Stat
          value="12.6 μs"
          label={t({ zh: "KDA 每 token 成本，收敛值", en: "KDA cost per input token, converged" })}
          tone="success"
        />
        <Stat
          value="13.8k tok/s"
          label={t({ zh: "prefill 吞吐峰值 (8K)", en: "peak prefill throughput (at 8K)" })}
        />
      </Grid>

      <Grid columns={2} gap={24}>
        <Stack gap={8}>
          <H3>{t({ zh: "Prefill 时间构成", en: "Composition of prefill time" })}</H3>
          <Text size="small" tone="tertiary">
            {t({
              zh: "纵轴：占 prefill 计算时间的百分比 (%)，已剔除集合通信 · 横轴：输入长度 (tokens)",
              en: "y: percent of prefill compute time (%), collectives excluded · x: input length (tokens)",
            })}
          </Text>
          <BarChart
            categories={PF_CTX}
            series={[
              { name: t({ zh: "KDA (69 层)", en: "KDA (69 layers)" }), data: PF_KDA_PCT, tone: "info" },
              { name: t({ zh: "全注意力 MLA (24 层)", en: "Full attention MLA (24 layers)" }), data: PF_FULL_PCT, tone: "danger" },
              { name: t({ zh: "MoE FFN (92 层)", en: "MoE FFN (92 layers)" }), data: PF_MOE_PCT, tone: "warning" },
              { name: t({ zh: "残差库 / norm / 其他", en: "Residual bank / norm / other" }), data: PF_OTHER_PCT, tone: "neutral" },
            ]}
            stacked
            height={250}
            valueSuffix="%"
          />
        </Stack>
        <Stack gap={8}>
          <H3>{t({ zh: "每输入 token 的成本：线性 vs 超线性", en: "Cost per input token: linear vs superlinear" })}</H3>
          <Text size="small" tone="tertiary">
            {t({
              zh: "纵轴：每个输入 token 的 GPU 时间 (μs/token) · 横轴：输入长度 (tokens)",
              en: "y: GPU time per input token (μs/token) · x: input length (tokens)",
            })}
          </Text>
          <LineChart
            categories={PF_CTX}
            series={[
              { name: t({ zh: "KDA", en: "KDA" }), data: PF_KDA_PER_TOK, tone: "info" },
              { name: t({ zh: "全注意力 MLA", en: "Full attention MLA" }), data: PF_FULL_PER_TOK, tone: "danger" },
              { name: t({ zh: "MoE FFN", en: "MoE FFN" }), data: PF_MOE_PER_TOK, tone: "warning" },
            ]}
            height={250}
            valueSuffix=" μs"
            showValues
          />
          <Text size="small" tone="tertiary">
            {t({
              zh: "KDA 和 MoE 的每 token 成本收敛到常数（12.6 和 24.2 μs）——它们是线性的。全注意力先降后升，到 32K 反弹到 30.1 μs/token，因为它是二次的。",
              en: "KDA and MoE converge to a constant per-token cost (12.6 and 24.2 μs) — both are linear. Full attention falls and then turns back up, reaching 30.1 μs/token at 32K, because it is quadratic.",
            })}
          </Text>
        </Stack>
      </Grid>

      <Callout
        tone="warning"
        title={t({ zh: "Prefill 里的交叉点在 8K 附近", en: "The prefill crossover sits near 8K" })}
      >
        {t({
          zh: "按单层算，1K 和 4K 时一层全注意力反而比一层 KDA 便宜（0.77×、0.78×），8K 时基本打平（0.95×），到 32K 才变成 6.8×。也就是说在 8K 以下的 prefill 上，线性注意力并没有省下时间——69 层 KDA 花掉 22.7% 而 24 层全注意力只花 7.5%，纯粹是因为 KDA 层多了三倍。",
          en: "Per layer, a full-attention layer is actually cheaper than a KDA layer at 1K and 4K (0.77× and 0.78×), roughly level at 8K (0.95×), and only becomes 6.8× more expensive at 32K. Below 8K of prefill, linear attention buys no time at all: 69 KDA layers take 22.7% while 24 full-attention layers take 7.5%, purely because there are three times as many of them.",
        })}
      </Callout>

      <Table
        headers={[
          t({ zh: "输入长度", en: "Input length" }),
          t({ zh: "一层 KDA", en: "One KDA layer" }),
          t({ zh: "一层全注意力", en: "One full-attn layer" }),
          t({ zh: "比值 全注意力:KDA", en: "Ratio, full-attn : KDA" }),
          t({ zh: "谁更贵", en: "Which is dearer" }),
        ]}
        rows={PF_CTX.map((c, i) => [
          <Text weight="semibold">{c}</Text>,
          `${fmt(PF_KDA_PER_LAYER[i], 3)} ms`,
          `${fmt(PF_FULL_PER_LAYER[i], 3)} ms`,
          `${fmt(PF_LAYER_RATIO[i])}×`,
          PF_LAYER_RATIO[i] < 1
            ? t({ zh: "KDA 更贵", en: "KDA costs more" })
            : t({ zh: "全注意力更贵", en: "full attention costs more" }),
        ])}
        columnAlign={["left", "right", "right", "right", "left"]}
        rowTone={PF_LAYER_RATIO.map((r) => (r < 1 ? "warning" : undefined))}
        striped
      />
      <Text size="small" tone="tertiary">
        {t({
          zh: "每层 GPU 时间 = 该类型全部层在整个 prefill 中的时间 ÷ 层数（69 或 24），已剔除集合通信。",
          en: "Per-layer GPU time = time for all layers of that type over the whole prefill, divided by the layer count (69 or 24), collectives excluded.",
        })}
      </Text>

      <Stack gap={8}>
        <H3>{t({ zh: "Prefill 完整数值", en: "Full prefill numbers" })}</H3>
        <Table
          headers={[
            t({ zh: "输入长度", en: "Input length" }),
            t({ zh: "TTFT", en: "TTFT" }),
            t({ zh: "吞吐", en: "Throughput" }),
            t({ zh: "GPU 时间", en: "GPU time" }),
            t({ zh: "GPU忙/TTFT", en: "GPU busy / TTFT" }),
            t({ zh: "KDA", en: "KDA" }),
            t({ zh: "全注意力", en: "Full attn" }),
            t({ zh: "MoE", en: "MoE" }),
            t({ zh: "集合通信", en: "Collectives" }),
          ]}
          rows={PF_CTX.map((c, i) => [
            <Text weight="semibold">{c}</Text>,
            `${PF_TTFT_MS[i]} ms`,
            `${PF_TOK_PER_S[i].toLocaleString()} tok/s`,
            `${fmt(PF_TOTAL_GPU[i], 1)} ms`,
            fmt(PF_BUSY_OVER_TTFT[i]),
            `${fmt(PF_KDA[i], 1)} ms (${fmt(PF_KDA_PCT[i], 1)}%)`,
            `${fmt(PF_FULL[i], 1)} ms (${fmt(PF_FULL_PCT[i], 1)}%)`,
            `${fmt(PF_MOE[i], 1)} ms (${fmt(PF_MOE_PCT[i], 1)}%)`,
            `${fmt(PF_COLLECTIVE[i], 1)} ms`,
          ])}
          columnAlign={["left", "right", "right", "right", "right", "right", "right", "right", "right"]}
          striped
        />
        <Text size="small" tone="tertiary">
          {t({
            zh: "百分比是占计算时间（已剔除集合通信）的比例，与解码那张表口径一致。集合通信单列：它在 32K 时是 429 ms，占 GPU 时间的 14%，是真实的 all-reduce 成本。",
            en: "Percentages are shares of compute time with collectives excluded, matching the decode table. Collectives are listed separately: at 32K they are 429 ms, 14% of GPU time, and that is genuine all-reduce cost.",
          })}
        </Text>
      </Stack>

      <Stack gap={8}>
        <H3>{t({ zh: "承担各自数学的那一个核函数", en: "The one kernel that carries each mechanism" })}</H3>
        <Text size="small" tone="tertiary">
          {t({
            zh: "纵轴：整个 prefill 中该核函数的 GPU 时间 (ms) · 横轴：输入长度 (tokens) · MLA 是 _fwd_kernel，KDA 是 chunk_gated_delta_rule / chunk_kda / chunk_gla 三个分块核函数之和",
            en: "y: GPU time of that kernel over the whole prefill (ms) · x: input length (tokens) · MLA is _fwd_kernel; KDA is the sum of the chunk_gated_delta_rule, chunk_kda and chunk_gla chunked kernels",
          })}
        </Text>
        <BarChart
          categories={PF_CTX}
          series={[
            { name: t({ zh: "MLA prefill 注意力 (_fwd_kernel)", en: "MLA prefill attention (_fwd_kernel)" }), data: PF_MLA_KERNEL, tone: "danger" },
            { name: t({ zh: "KDA 分块线性注意力", en: "KDA chunked linear attention" }), data: PF_KDA_CHUNK_KERNEL, tone: "info" },
          ]}
          height={230}
          valueSuffix=" ms"
          showValues
        />
        <Text size="small" tone="tertiary">
          {t({
            zh: "从 8K 到 32K，输入变成 4 倍，MLA 的注意力核函数从 18.5 ms 涨到 916.1 ms（约 50 倍），而 KDA 的分块核函数只从 41.2 ms 涨到 176.9 ms（约 4.3 倍，即线性）。这一个核函数在 32K 时就占了整个 prefill GPU 时间的 30%。",
            en: "From 8K to 32K the input grows 4× while the MLA attention kernel goes from 18.5 ms to 916.1 ms, about 50×; the KDA chunked kernels go from 41.2 ms to 176.9 ms, about 4.3× — linear. That single MLA kernel is 30% of all prefill GPU time at 32K.",
          })}
        </Text>
      </Stack>
    </Stack>
  );
}

function ChunkSplit() {
  const { lang, t } = useT();
  return (
    <Stack gap={12}>
      <H2>{t({ zh: "那 50 倍是怎么来的：两个 chunk 跑的不是同一种 MLA", en: "Where that 50x comes from: the two chunks do not run the same MLA" })}</H2>
      {lang === "zh" ? (
        <Text tone="secondary">
          32K 的 prefill 是两个 16384 的 chunk。 把 trace 按前向边界切开逐核函数对比， 除注意力之外<b>每个核函数都一样</b>——
          KDA 的分块核 68.84 对 68.88 ms， MoE 的归约 34.72 对 34.73 ms。 只有 chunk 2 独有的几个核函数， 加起来 8.9 ms，
          占两者 783 ms 差距的 1%。 差距全部在同一个核函数、同样 24 次派发里。
        </Text>
      ) : (
        <Text tone="secondary">
          The 32K prefill runs as two chunks of 16384. Splitting the trace at the forward-pass
          boundary and comparing kernel by kernel, <b>everything except attention is identical</b> —
          KDA's chunk kernel 68.84 versus 68.88 ms, MoE's reduction 34.72 versus 34.73 ms. The
          kernels unique to chunk 2 total 8.9 ms, 1% of the 783 ms gap. All of it sits in one
          kernel at the same dispatch count.
        </Text>
      )}
      <Table
        headers={[
          "",
          t({ zh: "chunk 1（无前缀）", en: "chunk 1 (no prefix)" }),
          t({ zh: "chunk 2（16K 前缀）", en: "chunk 2 (16K prefix)" }),
          t({ zh: "比值", en: "ratio" }),
        ]}
        rows={[
          [t({ zh: "MLA 形式", en: "MLA form" }),
            t({ zh: "解压后的 MHA", en: "decompressed MHA" }),
            t({ zh: "吸收后的 latent", en: "absorbed latent" }), ""],
          ["Lq / Lv", CHUNK_SPLIT.lq[0], CHUNK_SPLIT.lq[1], "3.40×"],
          [t({ zh: "BLOCK_M / 启动 grid", en: "BLOCK_M / launch grid" }),
            `128 · ${CHUNK_SPLIT.gridZ[0]}`, `64 · ${CHUNK_SPLIT.gridZ[1]}`, ""],
          [t({ zh: "query-key 对数", en: "query-key pairs" }),
            "1.34e8", "4.03e8", "3.00×"],
          [t({ zh: "每层 FLOP", en: "FLOP per layer" }), "1.03e12", "1.05e13", "10.2×"],
          [t({ zh: "每层耗时", en: "ms per layer" }),
            `${fmt(CHUNK_SPLIT.msPerLayer[0])} ms`, `${fmt(CHUNK_SPLIT.msPerLayer[1])} ms`, "12.8×"],
          [t({ zh: "达到的算力", en: "achieved throughput" }),
            "373 TFLOP/s", "297 TFLOP/s", "0.80×"],
        ]}
        columnAlign={["left", "right", "right", "right"]}
        rowTone={[undefined, undefined, undefined, undefined, "warning", "warning", undefined]}
        striped
      />
      <Callout tone="warning" title={t({ zh: "更正一处早先的说法", en: "Correcting an earlier reading" })}>
        {t({
          zh: "先前把 8K→32K 超出二次的增长记在「核函数效率」上， 那是错的： 它默认两个 chunk 跑同一种数学。 实际上 chunk 2 做了 10.2 倍的算术， 效率只低 1.26 倍。 核函数没问题， 贵的是形式的选择。",
          en: "An earlier reading blamed the super-quadratic 8K→32K growth on kernel efficiency. That was wrong: it assumed both chunks ran the same maths. Chunk 2 does 10.2× the arithmetic at only 1.26× lower efficiency. The kernel is fine; the cost is the choice of form.",
        })}
      </Callout>
      {lang === "zh" ? (
        <Text>
          为什么会切换？ 一旦存在前缀， MLA 就换成吸收形式—— 直接对 576 维的 latent 做注意力， 省掉把前缀 KV 解压成
          每头 320 维的那一步。 对 decode 这是对的： 一个 query 对很多 key， 解压的成本摊不掉。 但对一个有 16384 个 query 的
          prefill chunk， 这个取舍反了： 为了省一次能被 16384 个 query 摊薄的解压， 每个 query-key 对多付 3.4 倍的 FLOP。
          SGLang 里治这个的功能叫 chunked prefix cache， 它让前缀那部分用解压形式跑—— 而它在
          <Code>triton</Code> 后端上不可用（只支持 flashinfer / fa3 / fa4 / flashmla / cutedsl_mla / cutlass_mla），
          所以这套配置只能付这个代价。
        </Text>
      ) : (
        <Text>
          Why the switch? Once a prefix exists, MLA moves to the absorbed form — attending
          directly over the 576-dim latent, which avoids decompressing the prefix KV into 320
          dims per head. For decode that is correct: one query against many keys, so the
          decompression can never be amortized. For a prefill chunk with 16384 queries the
          trade inverts — you pay 3.4× the FLOPs per query-key pair to avoid a decompression
          that 16384 queries would have amortized easily. The feature that fixes this is
          chunked prefix cache, which runs the prefix part in the decompressed form, and it is
          unavailable on the <Code>triton</Code> backend (only flashinfer / fa3 / fa4 /
          flashmla / cutedsl_mla / cutlass_mla), so this configuration simply pays.
        </Text>
      )}
      <Callout tone="info" title={t({ zh: "在 Perfetto 里为什么看不见", en: "Why you cannot see this in Perfetto" })}>
        {t({
          zh: `在那份 trace 里注意力占 GPU 时间的 ${ATTN_TIME_SHARE}%， 却只占切片数量的 ${ATTN_SLICE_SHARE}%——8727 个核函数里只有 48 个。 而且 92.8% 的注意力时间集中在后 24 个切片上， 位于 3.06 秒 trace 的 t≈1.17–3.05 秒。 在前 1.14 秒（chunk 1）里放大， 注意力只占那段窗口的 6.1%。 要按总时长排序， 不能靠眼睛看密度。下一节把这件事量化到底。`,
          en: `Attention is ${ATTN_TIME_SHARE}% of GPU time in that trace but only ${ATTN_SLICE_SHARE}% of its slices — 48 out of 8727. And 92.8% of the attention time is in the later 24 of them, between t≈1.17 s and 3.05 s of a 3.06 s trace. Zoom into the first 1.14 s — chunk 1 — and attention is 6.1% of that window. Sort by total duration; do not judge by visual density. The next section quantifies this.`,
        })}
      </Callout>
    </Stack>
  );
}

function PerfettoCrossCheck() {
  const { lang, t } = useT();
  return (
    <Stack gap={14}>
      <H2>
        {t({
          zh: "用 Perfetto 复核：数字是对的，看错的是那条轨道",
          en: "Cross-checked in Perfetto: the numbers hold, the wrong track was being read",
        })}
      </H2>
      {lang === "zh" ? (
        <Text tone="secondary">
          在 Perfetto 里看 32K 的 trace， 全注意力显得微不足道， 和这一页说的 37.6% 对不上。
          把同一份 trace 用三条互不共享代码的路径重算—— Perfetto 自己的 <Code>trace_processor</Code>、
          原始 chrome JSON 直读、 以及 PyTorch 自己投影到设备时间轴上的 <Code>gpu_user_annotation</Code> 区间——
          <b>三条路径都给出同一个数</b>。 分歧不在数据， 在于 Perfetto 里有两条都标着 <Code>K3/*</Code> 的轨道，
          而它们给出的答案正好相反。
        </Text>
      ) : (
        <Text tone="secondary">
          Opened in Perfetto, the 32K trace makes full attention look negligible, which does not
          match the 37.6% on this page. Recomputing the same trace along three paths that share no
          code — Perfetto's own <Code>trace_processor</Code>, a direct re-read of the chrome JSON,
          and the <Code>gpu_user_annotation</Code> ranges PyTorch itself projects onto the device
          timeline — <b>all three land on the same number</b>. The disagreement is not in the data.
          It is that Perfetto shows two tracks both labelled <Code>K3/*</Code>, and they answer
          opposite questions.
        </Text>
      )}

      <Grid columns={3} gap={16}>
        <Stat
          value="2.5%"
          label={t({
            zh: "K3/full_attn 占 CPU 轨道上 K3 区间的时长",
            en: "K3/full_attn share of the CPU track's K3 ranges",
          })}
          tone="danger"
        />
        <Stat
          value="39.4%"
          label={t({
            zh: "同一个区间占 GPU 轨道的时长",
            en: "the same range's share of the GPU track",
          })}
          tone="success"
        />
        <Stat
          value="0.03%"
          label={t({
            zh: "核函数区间重叠率（求和不会重复计数）",
            en: "kernel interval overlap (summing does not double count)",
          })}
        />
      </Grid>

      <Stack gap={8}>
        <H3>
          {t({
            zh: "同样四个区间，主机侧和设备侧的答案正好相反",
            en: "The same four ranges, measured on the host and on the device",
          })}
        </H3>
        <Table
          headers={[
            t({ zh: "record_function 区间", en: "record_function range" }),
            t({ zh: "CPU 线程 (sglang::scheduler_TP0)", en: "CPU thread (sglang::scheduler_TP0)" }),
            t({ zh: "占比", en: "share" }),
            t({ zh: "GPU 设备时间", en: "GPU device time" }),
            t({ zh: "占比", en: "share" }),
          ]}
          rows={TRACK_INVERSION.map((r) => [
            <Code>{r.block}</Code>,
            `${fmt(r.cpuMs, 1)} ms`,
            `${fmt((100 * r.cpuMs) / CPU_TRACK_TOTAL, 1)}%`,
            `${fmt(r.gpuMs, 1)} ms`,
            `${fmt((100 * r.gpuMs) / GPU_TRACK_TOTAL, 1)}%`,
          ])}
          columnAlign={["left", "right", "right", "right", "right"]}
          rowTone={TRACK_INVERSION.map((r) =>
            r.block === "K3/full_attn" || r.block === "K3/kda" ? "warning" : undefined,
          )}
          striped
        />
        {lang === "zh" ? (
          <Text size="small" tone="tertiary">
            CPU 轨道上 KDA 占 85.3%、全注意力占 2.5%； GPU 轨道上正好换过来。 原因是 prefill 阶段主机跑在设备前面，
            队列一满就阻塞在当时恰好打开的那个区间里—— 这份 trace 的 GPU 在 3.06 秒里忙了 {GPU_BUSY}%， 所以主机几乎一直在等。
            主机侧区间的宽度衡量的是「在哪里被堵住」， 不是「花了多少算力」。 本页所有构成数字都用设备侧核函数时长，
            从来没用过主机侧区间宽度。
          </Text>
        ) : (
          <Text size="small" tone="tertiary">
            On the CPU track KDA is 85.3% and full attention 2.5%; on the GPU track it is the other
            way round. The host runs ahead of the device during prefill and blocks wherever the
            launch queue happens to fill — the GPU in this trace is busy {GPU_BUSY}% of its 3.06 s
            span, so the host is stalled almost throughout. Host-side range width measures where it
            stalled, not what cost compute. Every composition number on this page uses device-side
            kernel duration; none uses host range width.
          </Text>
        )}
      </Stack>

      <Stack gap={8}>
        <H3>
          {t({
            zh: "就算看对了轨道，两个 chunk 也长得不一样",
            en: "Even on the right track, the two chunks do not look alike",
          })}
        </H3>
        <Table
          headers={[
            "",
            `chunk 1 · ${CHUNK_WINDOW[0]}`,
            `chunk 2 · ${CHUNK_WINDOW[1]}`,
          ]}
          rows={[
            ...CHUNK_COMPOSITION.map((r) => [
              t(r.block),
              `${fmt(r.c1, 1)} ms  (${fmt((100 * r.c1) / CHUNK_GPU_MS[0], 1)}%)`,
              `${fmt(r.c2, 1)} ms  (${fmt((100 * r.c2) / CHUNK_GPU_MS[1], 1)}%)`,
            ]),
            [
              t({ zh: "该 chunk 核函数总时间", en: "kernel time in the chunk" }),
              `${fmt(CHUNK_GPU_MS[0], 1)} ms`,
              `${fmt(CHUNK_GPU_MS[1], 1)} ms`,
            ],
            [
              t({ zh: "_fwd_kernel 占该窗口墙钟", en: "_fwd_kernel share of that window" }),
              `${fmt(CHUNK_ATTN_DENSITY[0], 1)}%`,
              `${fmt(CHUNK_ATTN_DENSITY[1], 1)}%`,
            ],
          ]}
          columnAlign={["left", "right", "right"]}
          rowTone={["warning", undefined, undefined, undefined, "warning"]}
          striped
        />
        {lang === "zh" ? (
          <Text size="small" tone="tertiary">
            MoE 和 KDA 两个 chunk 几乎一模一样（526.5 对 524.1 ms、276.8 对 272.1 ms）—— 它们对 token 数线性，
            而两个 chunk 都是 16384 个 token。 只有注意力变了。 所以 37.6% 是两个 chunk 的平均：
            滚到 trace 开头看到的是 10.6%， 滚到后面才是 47.6%。
          </Text>
        ) : (
          <Text size="small" tone="tertiary">
            MoE and KDA are near-identical across the chunks (526.5 vs 524.1 ms, 276.8 vs 272.1 ms)
            — both are linear in token count and both chunks carry 16384 tokens. Only attention
            moves. So 37.6% is the average of the two: scroll to the start of the trace and you see
            10.6%, scroll to the end and you see 47.6%.
          </Text>
        )}
      </Stack>

      <CollapsibleSection
        title={t({
          zh: "三条独立路径的复核结果，以及 Perfetto 自己丢掉的两个事件",
          en: "The three independent paths, and the two events Perfetto itself drops",
        })}
      >
        <Stack gap={10}>
          <Table
            headers={[
              t({ zh: "量", en: "quantity" }),
              t({ zh: "已发布 (bucketize.py)", en: "published (bucketize.py)" }),
              t({ zh: "原始 JSON 直读", en: "raw JSON re-parse" }),
              t({ zh: "Perfetto trace_processor", en: "Perfetto trace_processor" }),
              t({ zh: "PyTorch 自己的 GPU 区间", en: "PyTorch's own GPU bands" }),
            ]}
            rows={VERIFY.map((r) => [t(r.q), r.pub, r.raw, r.tp, r.band])}
            columnAlign={["left", "right", "right", "right", "right"]}
            striped
          />
          {lang === "zh" ? (
            <Text size="small">
              最后一列值得单独说： <Code>gpu_user_annotation</Code> 是 PyTorch profiler 自己把
              <Code>record_function</Code> 区间投影到设备时间轴的结果， 和我们的 correlation-id 归因完全无关。
              这些区间横跨 1031.98 ms， 我们的归因累加出 1031.97 ms—— GPU 忙碌率 {GPU_BUSY}%，
              所以区间宽度和区间内核函数时长本来就该是同一个数。 改用「哪个区间盖住这个核函数」重做整张表，
              得到 37.57%， 已发布的是 37.56%。 逐核函数比对， 两种方法在 8961 个派发里只有 18.6 ms 分歧（0.6%），
              全部是 correlation 归到 other、区间归到某个块的边缘核函数。
            </Text>
          ) : (
            <Text size="small">
              The last column is worth separating out: <Code>gpu_user_annotation</Code> is the
              PyTorch profiler's own projection of the <Code>record_function</Code> ranges onto the
              device timeline, computed independently of our correlation-id walk. Those bands span
              1031.98 ms where our walk sums 1031.97 ms of kernels — at {GPU_BUSY}% GPU busy, a
              band's width and the kernels inside it have to be the same number. Reattributing every
              kernel by which band contains it and rebuilding the whole table gives 37.57% against
              the published 37.56%. Compared kernel by kernel, the two methods disagree on 18.6 ms
              out of 8961 dispatches (0.6%), all of it edge kernels the correlation walk leaves in{" "}
              <Code>other</Code> and the band claims for a block.
            </Text>
          )}
          <Callout
            tone="warning"
            title={t({
              zh: "别拿 Perfetto 的求和当总数",
              en: "Do not take Perfetto's sums as the totals",
            })}
          >
            {t({
              zh: "Perfetto 的 Chrome-JSON 导入器会静默丢弃无法在同一轨道上嵌套放置的切片： 8961 个派发里丢了 2 个， 其中一个正是 35.4 ms 的 _fwd_kernel。 所以它报 47 次 880.73 ms， 而真值是 48 次 916.14 ms（24 个 MLA 层 × 2 个 chunk）。 用它做查询和结构分析很好， 求总量要回原始 JSON。",
              en: "Perfetto's Chrome-JSON importer silently drops slices it cannot place by nesting on one track: 2 of 8961 dispatches here, one of them a 35.4 ms _fwd_kernel. So it reports 47 dispatches at 880.73 ms where the truth is 48 at 916.14 ms — 24 MLA layers times 2 chunks. Excellent for querying and structure; go back to the raw JSON for totals.",
            })}
          </Callout>
          {lang === "zh" ? (
            <Text size="small" tone="tertiary">
              另外核对了一件事： 把所有核函数区间取并集是 {fmt(GPU_UNION_MS, 2)} ms， 直接求和是{" "}
              {fmt(GPU_SUM_MS, 2)} ms， 差 0.034%。 也就是说 GPU 上几乎没有并发， 按时长求和不会重复计数——
              这正是本页所有构成数字的前提。 首末核函数跨度 {fmt(GPU_SPAN_MS, 2)} ms， GPU 忙碌率 {GPU_BUSY}%。
            </Text>
          ) : (
            <Text size="small" tone="tertiary">
              One more check: the union of all kernel intervals is {fmt(GPU_UNION_MS, 2)} ms against
              a naive sum of {fmt(GPU_SUM_MS, 2)} ms, a 0.034% difference. Almost nothing runs
              concurrently, so summing durations does not double count — which is the assumption
              every composition number on this page rests on. First to last kernel spans{" "}
              {fmt(GPU_SPAN_MS, 2)} ms, giving {GPU_BUSY}% GPU busy.
            </Text>
          )}
        </Stack>
      </CollapsibleSection>
    </Stack>
  );
}

function PrefillVsDecode() {
  const { lang, t } = useT();
  return (
    <Stack gap={10}>
      <H2>{t({ zh: "两个阶段放在一起看", en: "The two phases side by side" })}</H2>
      <Table
        headers={[
          t({ zh: "问题", en: "Question" }),
          t({ zh: "Prefill", en: "Prefill" }),
          t({ zh: "Decode", en: "Decode" }),
        ]}
        rows={[
          [
            t({ zh: "KDA 占该阶段时间", en: "KDA share of the phase" }),
            t({ zh: "23.6% (1K) → 15.8% (32K)", en: "23.6% (1K) → 15.8% (32K)" }),
            t({ zh: "17.1% (4K) → 8.8% (1M)", en: "17.1% (4K) → 8.8% (1M)" }),
          ],
          [
            t({ zh: "单层 全注意力 : KDA", en: "Per layer, full-attn : KDA" }),
            t({ zh: "0.77× (1K) → 6.8× (32K)", en: "0.77× (1K) → 6.8× (32K)" }),
            t({ zh: "2.5× (4K) → 17.3× (1M)", en: "2.5× (4K) → 17.3× (1M)" }),
          ],
          [
            t({ zh: "KDA 的绝对成本随上下文", en: "KDA absolute cost vs context" }),
            t({ zh: "线性增长（每 token 恒定 12.6 μs）", en: "grows linearly (flat 12.6 μs per token)" }),
            t({ zh: "完全不变（每步 3.5 ms）", en: "completely flat (3.5 ms per step)" }),
          ],
          [
            t({ zh: "瓶颈类型", en: "Bottleneck type" }),
            t({ zh: "算力受限，GPU 忙碌率 ≈ 1.00", en: "compute-bound, GPU busy ratio ≈ 1.00" }),
            t({ zh: "4K 权重受限 → 1M KV 带宽受限", en: "weight-bound at 4K → KV-bandwidth-bound at 1M" }),
          ],
          [
            t({ zh: "MoE 里的大头", en: "What dominates MoE" }),
            t({ zh: "被路由专家 GEMM（token 多，算得满）", en: "routed-expert GEMMs (many tokens, real work)" }),
            t({ zh: "共享专家 + 路由开销（专家 GEMM 仅 13.6%）", en: "shared experts + routing (expert GEMMs only 13.6%)" }),
          ],
          [
            t({ zh: "最该优化的地方", en: "Where to optimize" }),
            t({ zh: "让前缀 chunk 别用吸收形式（chunked prefix cache）", en: "stop the prefix chunk using the absorbed form (chunked prefix cache)" }),
            t({ zh: "MLA KV 扫描（仅达峰值带宽 20.6%）", en: "the MLA KV scan (only 20.6% of peak bandwidth)" }),
          ],
        ]}
        columnAlign={["left", "left", "left"]}
        striped
      />
      {lang === "zh" ? (
        <Text tone="secondary">
          结论是一致的，只是分界点不同：<b>KDA 在两个阶段都不是成本中心，但它也不是免费的</b>。
          真正决定 KDA 划不划算的是上下文长度——prefill 要到 8K 以上、decode 要到 4K 以上，
          一层 KDA 才比一层全注意力便宜。在那之前，把 3/4 的层换成 KDA 省下的是显存（KV cache），不是时间。
        </Text>
      ) : (
        <Text tone="secondary">
          The conclusion is the same in both phases, only the threshold moves:{" "}
          <b>KDA is never the cost centre, but it is not free either</b>. What decides whether
          KDA pays off is context length — a KDA layer becomes cheaper than a full-attention
          layer above about 8K in prefill and above about 4K in decode. Below those points,
          replacing three quarters of the layers with KDA buys memory (KV cache), not time.
        </Text>
      )}
    </Stack>
  );
}

function Takeaways() {
  const { lang, t } = useT();
  return (
    <Stack gap={12}>
      <H2>{t({ zh: "结论", en: "Takeaways" })}</H2>
      <Grid columns={3} gap={16}>
        <Card>
          <CardHeader>{t({ zh: "1 · KDA 不是问题", en: "1 · KDA is not the problem" })}</CardHeader>
          <CardBody>
            <Text size="small">
              {t({
                zh: "74% 的层数在 decode 只换来 9–17% 的时间、在 prefill 只换来 16–24%，而且 decode 的绝对耗时在 4K→1M 完全恒定（3.52–3.59 ms）。把 3/4 的注意力换成 KDA，在长上下文上确实拿到了它承诺的东西。",
                en: "74% of the layers account for only 9–17% of decode time and 16–24% of prefill time, and the absolute decode cost is constant from 4K to 1M (3.52–3.59 ms). Replacing three quarters of the attention layers with KDA does deliver what it promises at long context.",
              })}
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>{t({ zh: "2 · 剩下的 24 层是长上下文的成本中心", en: "2 · The remaining 24 layers are the long-context cost centre" })}</CardHeader>
          <CardBody>
            <Text size="small">
              {t({
                zh: "1M 时 24 层全注意力吃掉 52.9% 的解码时间，其中 83% 是 KV 扫描。单层成本是 KDA 层的 17.3 倍。任何 1M 级别的解码优化都应该从这 24 层入手。",
                en: "At 1M the 24 full-attention layers take 52.9% of decode time, 83% of which is the KV scan. Per layer they cost 17.3× a KDA layer. Any optimization aimed at 1M decode should start there.",
              })}
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>{t({ zh: "3 · 最大的一块空地", en: "3 · The biggest piece of open ground" })}</CardHeader>
          <CardBody>
            <Text size="small">
              {t({
                zh: "KV 扫描只跑到 8 TB/s 峰值的 20.6%：1M 时它自己就要 17.61 ms，占 40.10 ms 设备时间的 44%。把它推到峰值的 65% 会把这一项压到约 5.6 ms，等于从每步 38.75 ms 里拿掉 12 ms——这是单点收益最大的一项。",
                en: "The KV scan reaches only 20.6% of the 8 TB/s peak: at 1M it alone costs 17.61 ms, 44% of the 40.10 ms device total. Taking it to 65% of peak would cut that term to about 5.6 ms, removing 12 ms from a 38.75 ms decode step — the largest single win available.",
              })}
            </Text>
          </CardBody>
        </Card>
      </Grid>
      {lang === "zh" ? (
        <Text tone="secondary">
          换个角度看这件事：在 4K 上，解码时间的一半（50.8%）在 MoE，注意力两种加起来才 32%；
          到 1M，MoE 掉到 26.2%，注意力两种占到 61.7%。<b>这个模型的解码在 64K 附近从「权重受限」切换到「KV 受限」</b>
          ——4K→64K 每步设备时间只涨了 1.8 ms，而 64K→1M 涨了 17.6 ms。
        </Text>
      ) : (
        <Text tone="secondary">
          Seen from the other side: at 4K half of decode time (50.8%) is MoE and the two
          attention mechanisms together are just 32%. At 1M, MoE drops to 26.2% and attention
          takes 61.7%. <b>Decode in this model crosses over from weight-bound to KV-bound
          somewhere around 64K</b> — going from 4K to 64K adds only 1.8 ms of device time per
          step, while going from 64K to 1M adds 17.6 ms.
        </Text>
      )}
    </Stack>
  );
}

function Method() {
  const { lang, t } = useT();
  return (
    <Stack gap={8}>
      <H2>{t({ zh: "方法与注意事项", en: "Method and caveats" })}</H2>
      <CollapsibleSection
        title={t({ zh: "怎么把核函数归到块类型上", en: "How kernels were attributed to block types" })}
      >
        <Stack gap={8}>
          <Text size="small">
            {t({
              zh: "按核函数名字猜是不行的：Tensile 的 GEMM 名字只编码 tile 形状，KDA 的 q_proj 和共享专家的 GEMM 名字可能完全一样。所以先在 kimi_k3.py 的 KimiK3DecoderLayer 里加了可选的 record_function 区间（K3/kda、K3/full_attn、K3/moe），用 SGLANG_K3_PROF_RANGES 打开，跑一遍拿到每个核函数名到块类型的真值映射。",
              en: "Guessing from kernel names does not work: Tensile GEMM names encode tile shapes only, so a KDA q_proj and a shared-expert GEMM can carry identical names. Optional record_function ranges (K3/kda, K3/full_attn, K3/moe) were therefore added to KimiK3DecoderLayer in kimi_k3.py, enabled by SGLANG_K3_PROF_RANGES, and one run established the ground-truth map from kernel name to block type.",
            })}
          </Text>
          <Text size="small">
            {t({
              zh: "结果是 81% 的非集合通信派发的名字唯一属于一个块；唯一真正共享的计算核函数是 1536→7168 的 o_proj（KDA 和 MLA 各发 69 / 24 次，形状完全相同），按 69:24 的固定比例切分是精确的，不是近似。",
              en: "81% of non-collective dispatches have a name unique to one block. The only genuinely shared compute kernel is the 1536→7168 o_proj, which KDA and MLA issue 69 and 24 times respectively with identical shapes, so splitting it in a fixed 69:24 ratio is exact rather than approximate.",
            })}
          </Text>
          <Text size="small">
            {t({
              zh: "每个上下文点都会验证结构不变量：每步必须恰好 69 次 KDA 递推派发、24 次 MLA stage-1、92 次专家 GEMM，且每个已映射名字的每步派发次数必须是整数。5 个点全部通过，且没有未映射的核函数名。",
              en: "Every context point re-checks the structural invariants: exactly 69 KDA recurrence dispatches, 24 MLA stage-1 dispatches and 92 expert-GEMM dispatches per step, with an integral per-step count for every mapped name. All five points pass, with no unmapped kernel names.",
            })}
          </Text>
        </Stack>
      </CollapsibleSection>
      <Divider />
      <CollapsibleSection
        title={t({ zh: "为什么 TP8 集合通信的时间被排除", en: "Why the TP8 collective time is excluded" })}
      >
        <Stack gap={8}>
          <Text size="small">
            {t({
              zh: "torch profiler 看不到 HIP graph 内部重放的核函数，所以构成测量必须在关掉 CUDA graph 的 eager 模式下做。eager 模式下每步要发射 3372 个核函数，各 rank 的发射抖动使 aiter 的 all-reduce 核函数在自旋等待对端，187 次集合通信的「时长」累积到 73–121 ms/步——那是主机端抖动，不是通信成本，所以从构成里剔除。",
              en: "The torch profiler cannot see kernels replayed from inside a HIP graph, so the composition had to be measured in eager mode with CUDA graphs off. In eager mode each step issues 3372 kernels, and launch jitter across ranks leaves aiter's all-reduce kernels spinning on their peers: the 187 collectives per step accumulate 73–121 ms of apparent duration. That is host-side jitter, not communication cost, so it is excluded.",
            })}
          </Text>
          <Text size="small">
            {t({
              zh: "这个剔除是有校准的：同一配置打开 CUDA graph 后实测的每步延迟，与 eager 下的计算核函数时间之和的比值稳定在 0.93–0.97（4K 19.39 vs 20.77，32K 20.39 vs 21.79，64K 21.19 vs 22.53，512K 29.67 vs 31.16，1M 38.75 vs 40.10）。真实延迟略低于 eager 的核函数时间之和，说明真实的集合通信与空隙合起来接近于零净成本，构成可以直接搬到真实时间上。",
              en: "The exclusion is calibrated: with CUDA graphs enabled on the same configuration, measured per-step latency divided by the eager compute-kernel sum is a steady 0.93–0.97 (4K 19.39 vs 20.77, 32K 20.39 vs 21.79, 64K 21.19 vs 22.53, 512K 29.67 vs 31.16, 1M 38.75 vs 40.10). Real latency comes in slightly below the eager kernel sum, so real collectives plus gaps net out to roughly zero and the composition carries over to real time.",
            })}
          </Text>
        </Stack>
      </CollapsibleSection>
      <Divider />
      <CollapsibleSection title={t({ zh: "Prefill 是怎么测的，以及一个被自己的探针骗到的地方", en: "How prefill was measured, and one place the probe fooled itself" })}>
        <Stack gap={8}>
          <Text size="small">
            {t({
              zh: "每个 prefill 点只开一次 profiler：按 chunked-prefill 的 chunk 数（16384/chunk，所以 1K/4K/8K 各 1 个、32K 2 个）设定 forward 预算，武装后服务器处于空闲，随后立刻发请求，因此窗口里恰好只有那几个 prefill chunk；max_new_tokens=1 把 decode 挡在窗口外。区间直接归因，无需名字映射。",
              en: "One profiler session per prefill point. The forward budget is set to the number of chunked-prefill chunks (16384 per chunk, so one each for 1K/4K/8K and two for 32K); the server is idle when the profiler is armed and the request follows immediately, so the window holds exactly those chunks. A single output token keeps decode out. Ranges attribute the kernels directly, with no name map involved.",
            })}
          </Text>
          <Text size="small">
            {t({
              zh: "第一版结论说「短 prefill 是主机发射受限」，那是错的，而且错因就是探针本身：开着 record_function 区间时 4K 的 TTFT 是 692 ms，关掉后是 333 ms，而 32K 两者都是 ~3.05 s。区间对饱和的 GPU 可以忽略，对没饱和的不行。用不带区间的服务器重测后，四个点的「核函数时间和 / TTFT」分别是 1.00、0.99、0.99、1.00——prefill 在 1K 就已经把 GPU 跑满了。核函数时长不受区间影响，所以构成数字自始至终有效，被修正的只是墙钟那一列。",
              en: "A first pass concluded that short prefill was host-launch-bound. That was wrong, and the probe itself caused it: with the record_function ranges compiled in, 4K TTFT was 692 ms versus 333 ms without, while 32K was ~3.05 s either way. The ranges are negligible against a saturated GPU and not against an idle one. Re-measured on a server built without them, summed-kernel-time over TTFT is 1.00, 0.99, 0.99 and 1.00 across the four points — prefill already saturates the GPU at 1K. Kernel durations are unaffected by the ranges, so the composition numbers were valid throughout; only the wall-clock column needed correcting.",
            })}
          </Text>
          <Text size="small">
            {t({
              zh: "TTFT 取三次里的最小值，丢掉第一次冷调用（aiter 对未见过的 GEMM 形状要现场挑配置）。",
              en: "TTFT is the best of three, discarding the first cold call, since aiter selects a configuration on the fly for GEMM shapes it has not seen.",
            })}
          </Text>
        </Stack>
      </CollapsibleSection>
      <Divider />
      <CollapsibleSection title={t({ zh: "测量条件与边界", en: "Measurement conditions and limits" })}>
        <Stack gap={8}>
          <Text size="small">
            {t({
              zh: "批大小 1、无投机解码。这既是长上下文下的现实工作点（1M 上下文时 KV 显存基本只容得下一条序列），也去掉了 draft 模型与接受长度这两个混淆因素。线上部署那份配置额外开了 DSpark 投机解码（block_size 3），其 draft 模型是 5 层 Qwen3 风格稠密模型、不含 KDA 层。",
              en: "Batch size 1 with no speculative decoding. That is both the realistic operating point at long context (at 1M the KV cache barely fits one sequence) and a way to remove two confounds, the draft model and accept length. The deployed configuration additionally runs DSpark speculative decoding with block_size 3, whose draft model is a 5-layer Qwen3-style dense model containing no KDA layers.",
            })}
          </Text>
          <Text size="small">
            {t({
              zh: "所有构成数字取自 rank TP0，24 个连续解码步的平均值，profiler 在 prefill 结束、首 token 已产出之后才启动，因此窗口内只有稳态解码。1M 点的 prefill 用了 1462 秒（chunked prefill 16384，64 个 chunk）。",
              en: "All composition numbers come from rank TP0 as the mean of 24 consecutive decode steps. The profiler is armed only after prefill has produced its first token, so the window contains steady-state decode only. Prefill for the 1M point took 1462 s (chunked prefill of 16384, 64 chunks).",
            })}
          </Text>
          <Text size="small">
            {t({
              zh: "ATT 轨迹取自一个与服务器同形状的独立复现进程：整机 8 rank 下开 rocprofv3 逐派发拦截会把 scheduler 拖过 300 秒看门狗。因为 KDA 解码核函数的启动形状与上下文无关，这个替代是等价的，不是近似。默认 att-target-cu 抓不到波（48 个 workgroup 落不到目标 CU），需要 --att-shader-engine-mask 0xFF 配合 --att-consecutive-kernels。",
              en: "The ATT trace comes from a standalone reproduction at the server's exact shapes: enabling rocprofv3 per-dispatch interception across all 8 ranks drove the scheduler past its 300 s watchdog. Because the KDA decode kernel's launch geometry is context-independent, this substitution is equivalent rather than approximate. The default att-target-cu captures no waves — 48 workgroups never land on the target CU — so --att-shader-engine-mask 0xFF together with --att-consecutive-kernels is required.",
            })}
          </Text>
        </Stack>
      </CollapsibleSection>
    </Stack>
  );
}

export default function KimiK3KdaDecodeShare() {
  return (
    <Stack gap={28} style={{ padding: 24, maxWidth: 1180 }}>
      <Header />
      <Hero />
      <MainChart />
      <ShareAndPerLayer />
      <NumbersTable />
      <WhyDiverge />
      <Grid columns={1} gap={24}>
        <InsideKDA />
      </Grid>
      <Grid columns={2} gap={24}>
        <InsideFullAttn />
        <InsideMoE />
      </Grid>
      <AttTrace />
      <PrefillSection />
      <ChunkSplit />
      <PerfettoCrossCheck />
      <PrefillVsDecode />
      <Takeaways />
      <Method />
    </Stack>
  );
}
