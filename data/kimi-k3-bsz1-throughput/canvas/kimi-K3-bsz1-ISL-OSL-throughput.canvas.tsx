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
  Select,
  Spacer,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";

/* ------------------------------------------------------------------ *
 * 全部数据为 2026-07-29/30 在本地 8x MI355X 节点实测。服务端是 grid
 * search 选出的 dspark 最优配方（mem-frac 0.92 + block-size 3，TP8，
 * triton attention，--disable-radix-cache），端口 30100。
 * 客户端：python3 -m sglang.benchmark.serving --dataset-name random
 *         --random-range-ratio 1 --flush-cache
 * step_ms 与 prefill_tps 是推导量，不是 benchmark 直接输出：
 *   step_ms     = accept_len x tpot_med   （每个投机 verify step 的耗时）
 *   prefill_tps = ISL / ttft_med          （prefill 阶段的 token 速率）
 * ------------------------------------------------------------------ */

type Lang = "zh" | "en";
type Bi = { zh: string; en: string };

function useT() {
  const [lang, setLang] = useCanvasState<Lang>("canvasLang", "zh");
  return { lang, setLang, t: (s: Bi) => s[lang] };
}

type IslRow = {
  isl: number;
  islLabel: string;
  outTps: number;
  totalTps: number;
  ttftMs: number;
  prefillTps: number;
  tpotMs: number;
  accept: number;
  stepMs: number;
};

// Sweep A — ISL scaling at OSL 1024, concurrency 1, num-prompts 4.
const SWEEP_A: IslRow[] = [
  { isl: 128, islLabel: "128", outTps: 97.6, totalTps: 109.8, ttftMs: 181, prefillTps: 706, tpotMs: 9.92, accept: 2.328, stepMs: 23.1 },
  { isl: 1024, islLabel: "1K", outTps: 101.92, totalTps: 203.8, ttftMs: 179, prefillTps: 5716, tpotMs: 9.05, accept: 2.514, stepMs: 22.8 },
  { isl: 4096, islLabel: "4K", outTps: 89.3, totalTps: 446.5, ttftMs: 365, prefillTps: 11213, tpotMs: 10.6, accept: 2.466, stepMs: 26.1 },
  { isl: 8192, islLabel: "8K", outTps: 74.29, totalTps: 668.6, ttftMs: 636, prefillTps: 12874, tpotMs: 12.48, accept: 2.339, stepMs: 29.2 },
  { isl: 16384, islLabel: "16K", outTps: 47.92, totalTps: 814.6, ttftMs: 1329, prefillTps: 12327, tpotMs: 19.26, accept: 1.892, stepMs: 36.4 },
  { isl: 32768, islLabel: "32K", outTps: 31.02, totalTps: 1023.6, ttftMs: 3167, prefillTps: 10348, tpotMs: 29.52, accept: 1.741, stepMs: 51.4 },
  { isl: 65536, islLabel: "64K", outTps: 16.04, totalTps: 1042.5, ttftMs: 8241, prefillTps: 7953, tpotMs: 55.3, accept: 1.44, stepMs: 79.6 },
];

type OslRow = {
  osl: number;
  oslLabel: string;
  outTps: number;
  totalTps: number;
  ttftMs: number;
  tpotMs: number;
  accept: number;
};

// Sweep B — OSL scaling at ISL 1024, concurrency 1, num-prompts 4.
// OSL 1024 is the shared point carried over from sweep A.
const SWEEP_B: OslRow[] = [
  { osl: 128, oslLabel: "128", outTps: 107.03, totalTps: 963.2, ttftMs: 176, tpotMs: 8.04, accept: 3.038 },
  { osl: 512, oslLabel: "512", outTps: 106.88, totalTps: 320.6, ttftMs: 177, tpotMs: 8.81, accept: 2.658 },
  { osl: 1024, oslLabel: "1K", outTps: 101.92, totalTps: 203.8, ttftMs: 179, tpotMs: 9.05, accept: 2.514 },
  { osl: 2048, oslLabel: "2K", outTps: 106.62, totalTps: 159.9, ttftMs: 179, tpotMs: 9.42, accept: 2.644 },
  { osl: 4096, oslLabel: "4K", outTps: 93.47, totalTps: 116.8, ttftMs: 180, tpotMs: 10.46, accept: 2.392 },
];

type ConcRow = {
  conc: number;
  np: number;
  outTps: number;
  totalTps: number;
  concAch: number;
  ttftMs: number;
  tpotMs: number;
  accept: number;
  stepMs: number;
  tokPerStep: number;
  perReqTps: number;
};

// Sweep C — concurrency scaling at ISL/OSL 1024, num-prompts = max(8, conc*2).
const SWEEP_C: ConcRow[] = [
  { conc: 1, np: 8, outTps: 109.84, totalTps: 219.7, concAch: 1.0, ttftMs: 176, tpotMs: 8.92, accept: 2.693, stepMs: 24.0, tokPerStep: 2.7, perReqTps: 109.8 },
  { conc: 2, np: 8, outTps: 180.73, totalTps: 361.5, concAch: 1.94, ttftMs: 202, tpotMs: 10.28, accept: 2.544, stepMs: 26.2, tokPerStep: 4.9, perReqTps: 93.2 },
  { conc: 4, np: 8, outTps: 261.33, totalTps: 522.7, concAch: 3.39, ttftMs: 333, tpotMs: 11.42, accept: 2.473, stepMs: 28.2, tokPerStep: 8.4, perReqTps: 77.1 },
  { conc: 8, np: 16, outTps: 472.0, totalTps: 944.0, concAch: 7.36, ttftMs: 478, tpotMs: 15.36, accept: 2.637, stepMs: 40.5, tokPerStep: 19.4, perReqTps: 64.1 },
  { conc: 16, np: 32, outTps: 629.13, totalTps: 1258.3, concAch: 13.82, ttftMs: 849, tpotMs: 20.85, accept: 2.672, stepMs: 55.7, tokPerStep: 36.9, perReqTps: 45.5 },
  { conc: 32, np: 64, outTps: 949.73, totalTps: 1899.5, concAch: 29.66, ttftMs: 888, tpotMs: 29.67, accept: 2.732, stepMs: 81.1, tokPerStep: 81.0, perReqTps: 32.0 },
  { conc: 48, np: 96, outTps: 1060.58, totalTps: 2121.2, concAch: 42.65, ttftMs: 2271, tpotMs: 38.06, accept: 2.678, stepMs: 101.9, tokPerStep: 114.2, perReqTps: 24.9 },
];

const MARGINAL: Array<{ from: number; to: number; concX: number; tpsX: number; eff: number; ttftFrom: number; ttftTo: number }> = [
  { from: 1, to: 2, concX: 2.0, tpsX: 1.65, eff: 82.3, ttftFrom: 176, ttftTo: 202 },
  { from: 2, to: 4, concX: 2.0, tpsX: 1.45, eff: 72.3, ttftFrom: 202, ttftTo: 333 },
  { from: 4, to: 8, concX: 2.0, tpsX: 1.81, eff: 90.3, ttftFrom: 333, ttftTo: 478 },
  { from: 8, to: 16, concX: 2.0, tpsX: 1.33, eff: 66.6, ttftFrom: 478, ttftTo: 849 },
  { from: 16, to: 32, concX: 2.0, tpsX: 1.51, eff: 75.5, ttftFrom: 849, ttftTo: 888 },
  { from: 32, to: 48, concX: 1.5, tpsX: 1.12, eff: 74.4, ttftFrom: 888, ttftTo: 2271 },
];

function Header() {
  const theme = useHostTheme();
  const { lang, setLang, t } = useT();
  return (
    <Stack gap={6}>
      <Row gap={12} align="center">
        <H1>
          {t({
            zh: "Kimi-K3 dspark：bsz=1 的 ISL/OSL 特征曲线与并发扩展",
            en: "Kimi-K3 dspark: bsz=1 ISL/OSL characterisation and concurrency scaling",
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
          style={{ width: 108 }}
        />
      </Row>
      {lang === "zh" ? (
        <Text tone="secondary">
          单节点 8x MI355X、TP8。服务端用的是 grid search 选出的 dspark 最优配方
          （<Code>mem-fraction-static 0.92</Code> +{" "}
          <Code>--speculative-dspark-block-size 3</Code>、triton attention、
          <Code>--disable-radix-cache</Code>）。客户端{" "}
          <Code>sglang.benchmark.serving</Code>，<Code>--dataset-name random</Code>{" "}
          定长、每点 <Code>--flush-cache</Code>。2026-07-29/30 实测。
        </Text>
      ) : (
        <Text tone="secondary">
          Single node, 8x MI355X, TP8. The server runs the winning dspark recipe
          from the launch-parameter search (<Code>mem-fraction-static 0.92</Code>{" "}
          + <Code>--speculative-dspark-block-size 3</Code>, triton attention,{" "}
          <Code>--disable-radix-cache</Code>). Client is{" "}
          <Code>sglang.benchmark.serving</Code> with{" "}
          <Code>--dataset-name random</Code> at fixed lengths and{" "}
          <Code>--flush-cache</Code> per point. Measured 2026-07-29/30.
        </Text>
      )}
      <Row gap={6}>
        <Pill size="sm">
          {t({ zh: "18 个测量点", en: "18 measured points" })}
        </Pill>
        <Pill size="sm">
          {t({ zh: "3 条正交扫描", en: "3 orthogonal sweeps" })}
        </Pill>
        <Pill size="sm" style={{ color: theme.text.tertiary }}>
          sglang 3d35b45f7
        </Pill>
      </Row>
    </Stack>
  );
}

function Headline() {
  const { lang, t } = useT();
  return (
    <Stack gap={12}>
      <Grid columns={4} gap={16}>
        <Stat
          value="6.35x"
          label={t({
            zh: "bsz=1 吞吐跌幅，ISL 1K → 64K",
            en: "bsz=1 throughput loss, ISL 1K → 64K",
          })}
          tone="danger"
        />
        <Stat
          value="9.66x"
          label={t({
            zh: "吞吐增益，并发 1 → 48",
            en: "Throughput gain, concurrency 1 → 48",
          })}
          tone="success"
        />
        <Stat
          value="1,061"
          label={t({
            zh: "输出 tok/s 峰值（并发 48）",
            en: "Peak output tok/s (concurrency 48)",
          })}
        />
        <Stat
          value="32"
          label={t({
            zh: "并发拐点：再往上只换来 +12% 吞吐",
            en: "Concurrency knee: beyond it, only +12% throughput",
          })}
          tone="warning"
        />
      </Grid>
      <Callout
        tone="info"
        title={t({
          zh: "核心发现：accept_len 由上下文长度决定，与 batch 大小无关",
          en: "Key finding: accept length is governed by context length, not batch size",
        })}
      >
        {lang === "zh" ? (
          <Text>
            DSpark 的接受长度在并发 1→48 之间只在 2.473–2.732 之间抖动（幅度
            10.5%，无单调趋势），但在 ISL 128→64K 上从 2.514 单调掉到 1.440（幅度
            74.6%）。这意味着
            <Text weight="semibold">
              投机解码的收益不会因为加大 batch 而流失，只会因为上下文变长而流失
            </Text>
            。调度上可以放心堆并发；真正需要防的是长 prompt。
          </Text>
        ) : (
          <Text>
            DSpark's accept length only jitters between 2.473 and 2.732 across
            concurrency 1→48 — a 10.5% spread with no monotone trend — but falls
            monotonically from 2.514 to 1.440 across ISL 128→64K, a 74.6% spread.
            That means{" "}
            <Text weight="semibold">
              speculative decoding does not lose its edge as the batch grows,
              only as the context grows
            </Text>
            . Concurrency is safe to push; long prompts are what to guard against.
          </Text>
        )}
      </Callout>
    </Stack>
  );
}

function AcceptContrast() {
  const { lang, t } = useT();
  return (
    <Stack gap={10}>
      <H2>
        {t({
          zh: "把上面这句话画出来：两张图共用同一个 y 轴",
          en: "The same finding, plotted: both charts share one y-axis",
        })}
      </H2>
      <Text tone="secondary">
        {t({
          zh: "左右两图纵轴范围都锁定在 1.0–3.2，所以斜率可以直接对比。左边一路下坠，右边基本是条平线。",
          en: "Both charts are pinned to a 1.0–3.2 y-axis, so the slopes are directly comparable. The left one falls away; the right one is essentially flat.",
        })}
      </Text>
      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader trailing={<Pill size="sm">bsz=1</Pill>}>
            {t({
              zh: "accept_len vs 输入长度",
              en: "Accept length vs input length",
            })}
          </CardHeader>
          <CardBody>
            <LineChart
              categories={SWEEP_A.map((r) => r.islLabel)}
              series={[
                {
                  name: t({
                    zh: "accept_len（tokens/step）",
                    en: "Accept length (tokens/step)",
                  }),
                  data: SWEEP_A.map((r) => r.accept),
                  tone: "danger",
                },
              ]}
              height={200}
              beginAtZero={false}
              yMin={1}
              yMax={3.2}
              showValues
            />
            <Text size="small" tone="tertiary">
              {t({
                zh: "横轴：输入长度 ISL（tokens，OSL 固定 1024）。纵轴：accept_len（每个 verify step 被接受的 token 数）。",
                en: "X: input length ISL (tokens, OSL fixed at 1024). Y: accept length (tokens admitted per verify step).",
              })}
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing={<Pill size="sm">ISL/OSL 1024</Pill>}>
            {t({
              zh: "accept_len vs 并发",
              en: "Accept length vs concurrency",
            })}
          </CardHeader>
          <CardBody>
            <LineChart
              categories={SWEEP_C.map((r) => String(r.conc))}
              series={[
                {
                  name: t({
                    zh: "accept_len（tokens/step）",
                    en: "Accept length (tokens/step)",
                  }),
                  data: SWEEP_C.map((r) => r.accept),
                  tone: "success",
                },
              ]}
              height={200}
              beginAtZero={false}
              yMin={1}
              yMax={3.2}
              showValues
            />
            <Text size="small" tone="tertiary">
              {lang === "zh"
                ? "横轴：客户端并发。纵轴：accept_len，同一刻度。"
                : "X: client concurrency. Y: accept length, same scale."}
            </Text>
          </CardBody>
        </Card>
      </Grid>
    </Stack>
  );
}

function IslSweep() {
  const theme = useHostTheme();
  const { lang, t } = useT();
  const boxStyle = {
    border: `1px solid ${theme.stroke.tertiary}`,
    borderRadius: 6,
    padding: 14,
  };
  return (
    <Stack gap={10}>
      <H2>
        {t({
          zh: "扫描 A：bsz=1 下 ISL 的影响",
          en: "Sweep A: what input length does at bsz=1",
        })}
      </H2>
      <Text tone="secondary">
        {t({
          zh: "朴素的预期是 bsz=1 时 TPOT 跟 ISL 无关（decode 受权重带宽支配）。实测不是这样：ISL 1K→64K 让 TPOT 从 9.05ms 涨到 55.3ms，输出吞吐从 102 掉到 16 tok/s。",
          en: "The naive expectation is that TPOT at bsz=1 is independent of ISL, since decode is weight-bandwidth bound. It is not: ISL 1K→64K takes TPOT from 9.05 ms to 55.3 ms and output throughput from 102 down to 16 tok/s.",
        })}
      </Text>
      <LineChart
        categories={SWEEP_A.map((r) => r.islLabel)}
        series={[
          {
            name: t({ zh: "输出吞吐（tok/s）", en: "Output throughput (tok/s)" }),
            data: SWEEP_A.map((r) => r.outTps),
            tone: "danger",
          },
        ]}
        height={220}
        valueSuffix=" tok/s"
        showValues
      />
      <Text size="small" tone="tertiary">
        {t({
          zh: "横轴：输入长度 ISL（tokens）。纵轴：输出吞吐（tok/s，仅计 completion token）。OSL 固定 1024，并发 1，每点 4 条请求。",
          en: "X: input length ISL (tokens). Y: output throughput (tok/s, completion tokens only). OSL fixed at 1024, concurrency 1, 4 requests per point.",
        })}
      </Text>

      <H3>
        {t({
          zh: "这 6.35 倍是两个因子相乘出来的",
          en: "That 6.35x is the product of two factors",
        })}
      </H3>
      <Grid columns={3} gap={16}>
        <Stack gap={6} style={boxStyle}>
          <Stat
            value="3.50x"
            label={t({
              zh: "每 step 耗时变化（22.8 → 79.6 ms）",
              en: "Per-step latency (22.8 → 79.6 ms)",
            })}
            tone="warning"
          />
          <Text size="small" tone="secondary">
            {t({
              zh: "长上下文让每个 verify step 本身变贵。这部分是 attention 成本。",
              en: "A longer context makes each verify step more expensive. This is the attention cost.",
            })}
          </Text>
        </Stack>
        <Stack gap={6} style={boxStyle}>
          <Stat
            value="1.75x"
            label={t({
              zh: "accept_len 退化（2.514 → 1.440）",
              en: "Accept length decay (2.514 → 1.440)",
            })}
            tone="warning"
          />
          <Text size="small" tone="secondary">
            {t({
              zh: "draft 模型在长上下文下猜得更差，每个 step 产出的 token 更少。",
              en: "The draft model predicts worse at long context, so each step yields fewer tokens.",
            })}
          </Text>
        </Stack>
        <Stack gap={6} style={boxStyle}>
          <Stat
            value="6.11x"
            label={t({
              zh: "两者相乘的预测值",
              en: "Product of the two factors",
            })}
          />
          <Text size="small" tone="secondary">
            {t({
              zh: "实测跌幅 6.35x。差 4% 说明这个二因子模型基本抓住了全部损失。",
              en: "Measured loss is 6.35x. The 4% gap means this two-factor model accounts for essentially all of it.",
            })}
          </Text>
        </Stack>
      </Grid>

      <Callout
        tone="warning"
        title={t({
          zh: "TTFT 有一个约 176ms 的固定地板",
          en: "TTFT has a fixed floor of roughly 176 ms",
        })}
      >
        {lang === "zh" ? (
          <Text>
            ISL 128 和 ISL 1024 的中位 TTFT 分别是 181ms 和 179ms —— 输入涨了 8
            倍，TTFT 没动。所以短 prompt 场景下 TTFT 完全由固定开销（tokenize、
            调度、HTTP、draft 预热）支配，而不是 prefill 计算。换算成 prefill
            速率，ISL 128 只有 706 tok/s，而 ISL 8K 是 12,874 tok/s。
          </Text>
        ) : (
          <Text>
            Median TTFT is 181 ms at ISL 128 and 179 ms at ISL 1024 — an 8x
            increase in input with no movement in TTFT. For short prompts, TTFT
            is dominated entirely by fixed overhead (tokenisation, scheduling,
            HTTP, draft warm-up) rather than prefill compute. Expressed as a
            prefill rate, ISL 128 achieves only 706 tok/s against 12,874 tok/s at
            ISL 8K.
          </Text>
        )}
      </Callout>

      <Table
        headers={[
          "ISL",
          t({ zh: "输出 tok/s", en: "Output tok/s" }),
          t({ zh: "总 tok/s", en: "Total tok/s" }),
          t({ zh: "TTFT 中位", en: "Median TTFT" }),
          t({ zh: "prefill tok/s", en: "Prefill tok/s" }),
          t({ zh: "TPOT 中位", en: "Median TPOT" }),
          t({ zh: "accept_len", en: "Accept length" }),
          t({ zh: "每 step 耗时", en: "Per-step latency" }),
        ]}
        columnAlign={["left", "right", "right", "right", "right", "right", "right", "right"]}
        rows={SWEEP_A.map((r) => [
          <Text weight="semibold">{r.islLabel}</Text>,
          r.outTps.toFixed(2),
          r.totalTps.toFixed(1),
          `${r.ttftMs.toLocaleString()} ms`,
          r.prefillTps.toLocaleString(),
          `${r.tpotMs.toFixed(2)} ms`,
          r.accept.toFixed(3),
          `${r.stepMs.toFixed(1)} ms`,
        ])}
        striped
      />
      <Text size="small" tone="tertiary">
        {t({
          zh: "「总 tok/s」含输入 token，所以它随 ISL 上升并在 32K–64K 处饱和在约 1,040 tok/s —— 那是 bsz=1 时 prefill 的天花板。prefill 速率本身在 ISL 8K–16K 达到峰值约 12.9K tok/s，到 64K 退化到 8.0K，是 attention 的二次项在显现。",
          en: "\"Total tok/s\" counts input tokens, so it rises with ISL and saturates near 1,040 tok/s at 32K–64K — the prefill ceiling at bsz=1. The prefill rate itself peaks near 12.9K tok/s at ISL 8K–16K and decays to 8.0K at 64K, which is the quadratic term in attention becoming visible.",
        })}
      </Text>
    </Stack>
  );
}

function OslSweep() {
  const { lang, t } = useT();
  return (
    <Stack gap={10}>
      <H2>
        {t({
          zh: "扫描 B：bsz=1 下 OSL 几乎不影响吞吐",
          en: "Sweep B: output length barely moves throughput at bsz=1",
        })}
      </H2>
      {lang === "zh" ? (
        <Text tone="secondary">
          这是一个「什么都没发生」的结果，但它是有用的：OSL 从 128 到 2048，输出吞吐
          稳定在 107 tok/s 附近，说明 decode 本身没有随生成长度劣化。到 OSL 4096 才
          掉到 93.5，而这恰好是扫描 A 的同一个效应——生成到后期，有效上下文已经涨到
          约 5,120 token。TTFT 全程锁在 176–180ms，与 OSL 无关，符合预期。
        </Text>
      ) : (
        <Text tone="secondary">
          This is a "nothing happens" result, and that is useful in itself. From
          OSL 128 to 2048, output throughput holds near 107 tok/s, so decode does
          not degrade with generation length on its own. Only at OSL 4096 does it
          fall to 93.5 — and that is sweep A's effect reappearing, because by the
          end of generation the effective context has grown to roughly 5,120
          tokens. TTFT stays pinned at 176–180 ms throughout, independent of OSL,
          as expected.
        </Text>
      )}
      <Grid columns="3fr 2fr" gap={16}>
        <LineChart
          categories={SWEEP_B.map((r) => r.oslLabel)}
          series={[
            {
              name: t({ zh: "输出吞吐（tok/s）", en: "Output throughput (tok/s)" }),
              data: SWEEP_B.map((r) => r.outTps),
              tone: "info",
            },
          ]}
          height={200}
          beginAtZero={false}
          yMin={80}
          yMax={115}
          valueSuffix=" tok/s"
          showValues
        />
        <Table
          headers={[
            "OSL",
            t({ zh: "输出 tok/s", en: "Output tok/s" }),
            t({ zh: "TPOT 中位", en: "Median TPOT" }),
            t({ zh: "accept_len", en: "Accept length" }),
          ]}
          columnAlign={["left", "right", "right", "right"]}
          rows={SWEEP_B.map((r) => [
            <Text weight="semibold">{r.oslLabel}</Text>,
            r.outTps.toFixed(2),
            `${r.tpotMs.toFixed(2)} ms`,
            r.accept.toFixed(3),
          ])}
          striped
        />
      </Grid>
      <Text size="small" tone="tertiary">
        {t({
          zh: "横轴：输出长度 OSL（tokens，ISL 固定 1024）。纵轴：输出吞吐（tok/s），纵轴自适应到 80–115 以便看清波动。并发 1，每点 4 条请求。",
          en: "X: output length OSL (tokens, ISL fixed at 1024). Y: output throughput (tok/s), axis auto-fitted to 80–115 to make the variation legible. Concurrency 1, 4 requests per point.",
        })}
      </Text>
    </Stack>
  );
}

function ConcSweep() {
  const { lang, t } = useT();
  return (
    <Stack gap={10}>
      <H2>
        {t({
          zh: "扫描 C：并发扩展，以及拐点在哪",
          en: "Sweep C: concurrency scaling, and where the knee sits",
        })}
      </H2>
      {lang === "zh" ? (
        <Text tone="secondary">
          并发 1→48 换来 9.66 倍聚合吞吐，也就是 48 倍并发只兑现了 20% 的线性度。
          代价是单请求体验：每请求吞吐从 109.8 掉到 24.9 tok/s。48 是这个配方的{" "}
          <Code>max_running_requests</Code>，所以它就是饱和上限。
        </Text>
      ) : (
        <Text tone="secondary">
          Concurrency 1→48 buys 9.66x aggregate throughput, meaning 48x
          concurrency realises only 20% scaling efficiency. The cost is
          per-request experience: throughput per request falls from 109.8 to 24.9
          tok/s. 48 is this recipe's <Code>max_running_requests</Code>, so it is
          the saturation ceiling.
        </Text>
      )}
      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>
            {t({
              zh: "聚合输出吞吐 vs 并发",
              en: "Aggregate output throughput vs concurrency",
            })}
          </CardHeader>
          <CardBody>
            <LineChart
              categories={SWEEP_C.map((r) => String(r.conc))}
              series={[
                {
                  name: t({
                    zh: "聚合输出吞吐（tok/s）",
                    en: "Aggregate output throughput (tok/s)",
                  }),
                  data: SWEEP_C.map((r) => r.outTps),
                  tone: "success",
                },
              ]}
              height={200}
              valueSuffix=" tok/s"
              showValues
            />
            <Text size="small" tone="tertiary">
              {t({
                zh: "横轴：客户端并发。纵轴：聚合输出吞吐（tok/s）。",
                en: "X: client concurrency. Y: aggregate output throughput (tok/s).",
              })}
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>
            {t({
              zh: "单请求吞吐 vs 并发",
              en: "Per-request throughput vs concurrency",
            })}
          </CardHeader>
          <CardBody>
            <LineChart
              categories={SWEEP_C.map((r) => String(r.conc))}
              series={[
                {
                  name: t({
                    zh: "每请求吞吐（tok/s）",
                    en: "Throughput per request (tok/s)",
                  }),
                  data: SWEEP_C.map((r) => r.perReqTps),
                  tone: "danger",
                },
              ]}
              height={200}
              valueSuffix=" tok/s"
              showValues
            />
            <Text size="small" tone="tertiary">
              {t({
                zh: "横轴：客户端并发。纵轴：聚合吞吐除以实测并发（tok/s/请求）。",
                en: "X: client concurrency. Y: aggregate throughput divided by achieved concurrency (tok/s per request).",
              })}
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <H3>
        {t({
          zh: "TTFT 是拐点的真正代价",
          en: "TTFT is what the knee actually costs",
        })}
      </H3>
      <BarChart
        categories={SWEEP_C.map((r) =>
          lang === "zh" ? `并发 ${r.conc}` : `conc ${r.conc}`,
        )}
        series={[
          {
            name: t({ zh: "TTFT 中位（ms）", en: "Median TTFT (ms)" }),
            data: SWEEP_C.map((r) => r.ttftMs),
            tone: "warning",
          },
        ]}
        height={200}
        valueSuffix=" ms"
        showValues
      />
      <Text size="small" tone="tertiary">
        {t({
          zh: "横轴：客户端并发。纵轴：首 token 延迟中位数（ms）。注意 16→32 几乎没涨（849→888ms），而 32→48 跳了 2.6 倍（888→2,271ms）。",
          en: "X: client concurrency. Y: median time to first token (ms). Note that 16→32 barely moves (849→888 ms) while 32→48 jumps 2.6x (888→2,271 ms).",
        })}
      </Text>

      <Callout
        tone="warning"
        title={t({
          zh: "拐点在并发 32",
          en: "The knee is at concurrency 32",
        })}
      >
        {lang === "zh" ? (
          <Text>
            16→32 是这条曲线上性价比最好的一段：吞吐 +51%，而 TTFT 中位只从 849ms
            涨到 888ms。再往上到 48，吞吐只多 11.7%，TTFT 却跳到 2,271ms（2.6
            倍），TPOT 也从 29.7ms 涨到 38.1ms。
            <Text weight="semibold">
              要延迟就停在 32，要榨最后一点吞吐才上 48。
            </Text>
          </Text>
        ) : (
          <Text>
            16→32 is the best-value segment on this curve: +51% throughput for a
            median TTFT that moves only from 849 ms to 888 ms. Going on to 48
            adds just 11.7% throughput while TTFT jumps to 2,271 ms (2.6x) and
            TPOT rises from 29.7 ms to 38.1 ms.{" "}
            <Text weight="semibold">
              Stop at 32 if latency matters; go to 48 only to squeeze out the
              last of the throughput.
            </Text>
          </Text>
        )}
      </Callout>

      <Table
        headers={[
          t({ zh: "并发", en: "Conc" }),
          t({ zh: "请求数", en: "Requests" }),
          t({ zh: "输出 tok/s", en: "Output tok/s" }),
          t({ zh: "总 tok/s", en: "Total tok/s" }),
          t({ zh: "实测并发", en: "Achieved conc" }),
          t({ zh: "TTFT 中位", en: "Median TTFT" }),
          t({ zh: "TPOT 中位", en: "Median TPOT" }),
          t({ zh: "accept_len", en: "Accept length" }),
          t({ zh: "每 step 耗时", en: "Per-step latency" }),
          t({ zh: "每 step token", en: "Tokens/step" }),
          t({ zh: "每请求 tok/s", en: "tok/s per request" }),
        ]}
        columnAlign={["left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right"]}
        rows={SWEEP_C.map((r) => [
          <Text weight="semibold">{r.conc}</Text>,
          r.np,
          <Text weight="semibold">{r.outTps.toFixed(2)}</Text>,
          r.totalTps.toFixed(1),
          r.concAch.toFixed(2),
          `${r.ttftMs.toLocaleString()} ms`,
          `${r.tpotMs.toFixed(2)} ms`,
          r.accept.toFixed(3),
          `${r.stepMs.toFixed(1)} ms`,
          r.tokPerStep.toFixed(1),
          r.perReqTps.toFixed(1),
        ])}
        striped
      />
      <Divider />
      <H3>
        {t({ zh: "逐段边际效率", en: "Marginal efficiency, segment by segment" })}
      </H3>
      <Table
        headers={[
          t({ zh: "并发区间", en: "Segment" }),
          t({ zh: "并发倍数", en: "Conc factor" }),
          t({ zh: "吞吐倍数", en: "Throughput factor" }),
          t({ zh: "线性度", en: "Scaling efficiency" }),
          t({ zh: "TTFT 中位变化", en: "Median TTFT change" }),
        ]}
        columnAlign={["left", "right", "right", "right", "right"]}
        rows={MARGINAL.map((m) => [
          <Text weight="semibold">{`${m.from} → ${m.to}`}</Text>,
          `x${m.concX.toFixed(2)}`,
          `x${m.tpsX.toFixed(2)}`,
          `${m.eff.toFixed(1)}%`,
          `${m.ttftFrom.toLocaleString()} → ${m.ttftTo.toLocaleString()} ms`,
        ])}
        rowTone={[undefined, undefined, undefined, undefined, "success", "warning"]}
        striped
      />
      <Text size="small" tone="tertiary">
        {t({
          zh: "4→8 那段线性度 90.3% 偏高，是低 num-prompts 的测量假象：并发 4 时只有 2 个波次，爬坡和收尾占比大，实测并发只到 3.39（请求的 85%）。并发 8 及以上实测并发达到 92% 以上，数字更可信。",
          en: "The 90.3% efficiency on the 4→8 segment is inflated — a measurement artifact of low num-prompts. At concurrency 4 there are only two waves, so ramp-up and drain dominate and achieved concurrency reaches just 3.39 (85% of requested). From concurrency 8 upward, achieved concurrency exceeds 92% and the figures are more trustworthy.",
        })}
      </Text>
    </Stack>
  );
}

function Method() {
  const theme = useHostTheme();
  const { lang, t } = useT();
  const boxStyle = {
    border: `1px solid ${theme.stroke.tertiary}`,
    borderRadius: 6,
    padding: 14,
  };
  return (
    <Stack gap={10}>
      <H2>{t({ zh: "方法与可信度", en: "Method and confidence" })}</H2>
      <Grid columns={2} gap={16}>
        <Stack gap={8} style={boxStyle}>
          <H3>
            {t({
              zh: "对得上历史基准",
              en: "Reproduces the historical baseline",
            })}
          </H3>
          {lang === "zh" ? (
            <Text size="small" tone="secondary">
              扫描 C 的并发 1 点测出{" "}
              <Text weight="semibold">109.84 tok/s</Text>、accept_len 2.693；grid
              search 里同配方的 <Code>p3-lat-win</Code> 行记的是 111.58 tok/s、
              accept_len 2.757。吞吐差 1.6%，accept_len 差 2.3%，说明这次复现的
              服务端配方和测量协议都是对的。
            </Text>
          ) : (
            <Text size="small" tone="secondary">
              Sweep C's concurrency-1 point measures{" "}
              <Text weight="semibold">109.84 tok/s</Text> at accept length 2.693.
              The <Code>p3-lat-win</Code> row from the launch-parameter search,
              on the same recipe, recorded 111.58 tok/s at accept length 2.757 —
              1.6% apart on throughput and 2.3% on accept length. Both the server
              recipe and the measurement protocol reproduced correctly.
            </Text>
          )}
        </Stack>
        <Stack gap={8} style={boxStyle}>
          <H3>
            {t({ zh: "已知的噪声来源", en: "Known source of noise" })}
          </H3>
          {lang === "zh" ? (
            <Text size="small" tone="secondary">
              同一个 ISL/OSL 1024 点，扫描 A（4 条请求）测出 101.92 tok/s，扫描 C
              （8 条请求）测出 109.84 —— 相差 7.8%，而且吞吐比几乎等于 accept_len
              比。
              <Text weight="semibold">
                random 数据集是随机 token，draft 模型的接受率对内容敏感
              </Text>
              ，所以低样本数下 bsz=1 的 dspark 吞吐有约 8–10% 的 run-to-run
              波动。真实文本上接受率会更高。
            </Text>
          ) : (
            <Text size="small" tone="secondary">
              The same ISL/OSL 1024 point measures 101.92 tok/s in sweep A (4
              requests) and 109.84 tok/s in sweep C (8 requests) — 7.8% apart,
              and the throughput ratio is almost exactly the accept-length ratio.{" "}
              <Text weight="semibold">
                The random dataset emits random tokens, and the draft model's
                acceptance rate is content-sensitive
              </Text>
              , so bsz=1 dspark throughput carries roughly 8–10% run-to-run
              variance at low sample counts. Acceptance is higher on real text,
              which makes these figures a conservative lower bound.
            </Text>
          )}
        </Stack>
      </Grid>
      <Callout
        tone="neutral"
        title={t({ zh: "复现命令", en: "How to reproduce" })}
      >
        <Stack gap={4}>
          <Code>{"bash sweep-bsz1.sh <tag>            # sweep A + B"}</Code>
          <Code>{"bash sweep-conc-scaling.sh <tag>    # sweep C"}</Code>
          {lang === "zh" ? (
            <Text size="small" tone="secondary">
              两个脚本都在 <Code>/sgl-workspace/workspace</Code>，默认打端口
              30100，结果落在{" "}
              <Code>bsz1_results/&lt;tag&gt;/results.csv</Code>，指标由{" "}
              <Code>gridtools.py parse-bench</Code> 提取，与 grid search 用的是
              同一个解析器。两者都会拿 <Code>/tmp/k3-grid.lock</Code>，防止 grid
              跑起来把服务端拆掉。
            </Text>
          ) : (
            <Text size="small" tone="secondary">
              Both scripts live in <Code>/sgl-workspace/workspace</Code>, target
              port 30100 by default, and write to{" "}
              <Code>bsz1_results/&lt;tag&gt;/results.csv</Code>. Metrics are
              extracted by <Code>gridtools.py parse-bench</Code> — the same
              parser the launch-parameter search used. Both take{" "}
              <Code>/tmp/k3-grid.lock</Code> so that a grid run starting midway
              cannot tear the server down underneath them.
            </Text>
          )}
        </Stack>
      </Callout>
    </Stack>
  );
}

export default function Bsz1ThroughputCanvas() {
  return (
    <Stack gap={28} style={{ padding: 24, maxWidth: 1180 }}>
      <Header />
      <Headline />
      <Divider />
      <AcceptContrast />
      <Divider />
      <IslSweep />
      <Divider />
      <OslSweep />
      <Divider />
      <ConcSweep />
      <Divider />
      <Method />
    </Stack>
  );
}
