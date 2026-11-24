import type { Lang } from "@lib/utils";

type Text = Record<Lang, string>;

// Every count on the home page is a snapshot taken on this date.
// Library counts: the cards in public/sources/index.html.
// Pull request counts: gh search prs --author jhinpan --merged --repo <owner/repo>
export const AS_OF = "2026-11-23";

export const STATS: { value: number; label: Text }[] = [
  { value: 43, label: { en: "entries in the Library", zh: "Library 条目" } },
  {
    value: 8,
    label: { en: "experiments, raw data committed", zh: "个实验， 原始数据随文提交" },
  },
  { value: 48, label: { en: "merged PRs · ROCm/FlyDSL", zh: "merged PR · ROCm/FlyDSL" } },
  {
    value: 11,
    label: { en: "merged PRs · sgl-project/sglang", zh: "merged PR · sgl-project/sglang" },
  },
];

// Top to bottom is the path of one request. Each layer links to a reading of that layer.
export const STACK: { name: string; items: string; href: string }[] = [
  {
    name: "MODEL",
    items: "GLM-5.3 · DeepSeek-V4.1 Flash · Kimi-K3",
    href: "/sources/kimi-k3-kda-time-share.html",
  },
  { name: "SERVING", items: "SGLang · sgl-amd-mono", href: "/sources/sglang.html" },
  { name: "KERNEL", items: "FlyDSL · AITER", href: "/sources/flydsl.html" },
  { name: "COMPILER", items: "MLIR · LLVM", href: "/sources/flydsl-compiler-redesign.html" },
  { name: "SILICON", items: "MI355X · MI455X", href: "/sources/gcnasm.html" },
];

export type BarTone = "accent" | "teal" | "teal-soft" | "muted" | "hatch";

export type Measurement = {
  kicker: string;
  model: string;
  title: string;
  href: string;
  // `bars` widths are percentages of the largest value in the card.
  rows: { label: Text; value: string; bars: { width: number; tone: BarTone }[] }[];
  note: Text;
  takeaway: Text;
};

// Numbers are copied from each experiment's own summary; keep them in step with the linked page.
export const MEASUREMENTS: Measurement[] = [
  {
    kicker: "Experiment 007",
    model: "GLM-5.3",
    title: "PD × monokernel: GLM-5.3 on MI355X",
    href: "/sources/glm53-pd-monokernel-mi355x.html",
    rows: [
      {
        label: { en: "PD decode", zh: "PD decode" },
        value: "213",
        bars: [{ width: 44.9, tone: "muted" }],
      },
      {
        label: { en: "+ monokernels", zh: "+ monokernels" },
        value: "474",
        bars: [{ width: 100, tone: "accent" }],
      },
    ],
    note: {
      en: "tok/s per user, decode, concurrency 1. GSM8K unchanged.",
      zh: "每用户 decode 吞吐（tok/s）， 并发 1。 GSM8K 不变。",
    },
    takeaway: {
      en: "Two plain replicas on the same 16 GPUs still win on AgentX.",
      zh: "同样 16 张 GPU 上， 两个普通副本在 AgentX 上仍然更快。",
    },
  },
  {
    kicker: "Experiment 005",
    model: "DeepSeek-V4.1",
    title: "GB300 vs MI355X, one token at a time",
    href: "/sources/dsv41-gb300-vs-mi355x.html",
    rows: [
      {
        label: { en: "GB300", zh: "GB300" },
        value: "4.61",
        bars: [{ width: 50.4, tone: "teal" }],
      },
      {
        label: { en: "GB300, 1 stream", zh: "GB300, 1 stream" },
        value: "7.34",
        bars: [{ width: 80.2, tone: "teal-soft" }],
      },
      {
        label: { en: "MI355X", zh: "MI355X" },
        value: "9.15",
        bars: [{ width: 100, tone: "accent" }],
      },
    ],
    note: {
      en: "ms per DSpark verify cycle, BS=1.",
      zh: "每个 DSpark verify 周期的耗时（ms）， BS=1。",
    },
    takeaway: {
      en: "60% of the gap is stream concurrency, not kernel time.",
      zh: "差距的 60% 来自 stream 并发， 不是 kernel 耗时。",
    },
  },
  {
    kicker: "Experiment 002",
    model: "Kimi-K3",
    title: "Kimi-K3: three of every four layers are linear",
    href: "/sources/kimi-k3-kda-time-share.html",
    rows: [
      {
        label: { en: "share of layers", zh: "层数占比" },
        value: "74%",
        bars: [{ width: 74, tone: "teal" }],
      },
      {
        label: { en: "share of time", zh: "时间占比" },
        value: "9–24%",
        bars: [
          { width: 9, tone: "accent" },
          { width: 15, tone: "hatch" },
        ],
      },
    ],
    note: {
      en: "Decode and prefill GPU time by layer type, 1K–1M context on 8× MI355X.",
      zh: "按层类型统计 decode 和 prefill 的 GPU 时间， 1K–1M 上下文， 8× MI355X。",
    },
    takeaway: {
      en: "Most layers are not where the time goes.",
      zh: "大多数层并不是时间花掉的地方。",
    },
  },
];

export type Chip = { label: Text; count?: number; href?: string };

const mergedBy = (repo: string) =>
  `https://github.com/${repo}/pulls?q=is%3Apr+author%3Ajhinpan+is%3Amerged`;

export const THREADS: { tag: string; title: Text; body: Text; chips: Chip[] }[] = [
  {
    tag: "A · Kernels",
    title: { en: "Write the kernel, then prove it", zh: "先写 kernel， 再证明它是对的" },
    body: {
      en: "GPU kernels in FlyDSL for MI355X and MI455X, and the reviews that gate them before they reach a serving backend.",
      zh: "用 FlyDSL 为 MI355X 和 MI455X 写 GPU kernel， 并在它们进入 serving 后端之前做 review 把关。",
    },
    chips: [
      {
        label: { en: "ROCm/FlyDSL", zh: "ROCm/FlyDSL" },
        count: 48,
        href: mergedBy("ROCm/FlyDSL"),
      },
      { label: { en: "ROCm/aiter", zh: "ROCm/aiter" }, count: 6, href: mergedBy("ROCm/aiter") },
    ],
  },
  {
    tag: "B · Serving",
    title: { en: "Measure the model, not the kernel", zh: "测模型， 不只测 kernel" },
    body: {
      en: "Model-level performance in SGLang on AMD: GLM-5.3, DeepSeek-V4.1 Flash and Kimi-K3, from one request to a loaded server.",
      zh: "在 AMD 上做 SGLang 的模型级性能： GLM-5.3、DeepSeek-V4.1 Flash 和 Kimi-K3， 从单个请求到满载的服务。",
    },
    chips: [
      {
        label: { en: "sgl-project/sglang", zh: "sgl-project/sglang" },
        count: 11,
        href: mergedBy("sgl-project/sglang"),
      },
      {
        label: { en: "sgl-amd-mono", zh: "sgl-amd-mono" },
        count: 9,
        href: mergedBy("sgl-project/sgl-amd-mono"),
      },
    ],
  },
  {
    tag: "C · Agents for kernels",
    title: { en: "Let agents write, let a harness judge", zh: "让 agent 写， 让 harness 判" },
    body: {
      en: "A fleet that schedules agent kernel-optimization jobs on MI355X nodes, and a deterministic validator for the pull requests they open.",
      zh: "一个把 agent 的 kernel 优化任务调度到 MI355X 节点上的集群， 以及一个对它们提交的 PR 做确定性验证的 validator。",
    },
    chips: [
      { label: { en: "private repositories", zh: "私有仓库" } },
      {
        label: { en: "Paper reading 002", zh: "Paper reading 002" },
        href: "/sources/kernel-design-agents.html",
      },
    ],
  },
];

export type Shelf = "code" | "paper" | "tutorial" | "blog" | "experiments";

// Newest first. A `post` takes its title and date from the blog collection;
// a `source` is a page under public/sources, which carries no date metadata.
export type LatestItem =
  | { kind: "source"; shelf: Shelf; label: string; date: string; title: string; href: string }
  | { kind: "post"; shelf: Shelf; label: string; slug: string };

export const LATEST: LatestItem[] = [
  {
    kind: "source",
    shelf: "experiments",
    label: "Experiment 008",
    date: "2026-11-23",
    title: "One decision, one decider: redesigning the FlyDSL compiler for agents",
    href: "/sources/flydsl-compiler-redesign.html",
  },
  {
    kind: "source",
    shelf: "code",
    label: "Code 012",
    date: "2026-11-17",
    title: "270 Fences: The Compiler in the FlyDSL Kernel Loop",
    href: "/sources/flydsl-compiler-in-the-loop.html",
  },
  {
    kind: "source",
    shelf: "paper",
    label: "Paper 006",
    date: "2026-11-17",
    title: "Together Compiler",
    href: "/sources/together-compiler.html",
  },
  { kind: "post", shelf: "blog", label: "Blog", slug: "glm53-pd-monokernel" },
  {
    kind: "source",
    shelf: "experiments",
    label: "Experiment 006",
    date: "2026-10-26",
    title: "GLM-5.3 monokernel hill-climbing on MI355X: a retrospective",
    href: "/sources/glm53-agentx-hillclimb-mi355x.html",
  },
  { kind: "post", shelf: "tutorial", label: "Tutorial", slug: "unit-pass-model-fail" },
];

// Start year of each stage; degree names follow the About page.
export const TIMELINE: { year: string; place: Text; detail: string }[] = [
  {
    year: "2020",
    place: { en: "Shanghai Jiao Tong University", zh: "上海交通大学" },
    detail: "BE, Electrical and Computer Engineering",
  },
  {
    year: "2022",
    place: { en: "University of Michigan", zh: "密歇根大学" },
    detail: "BSE, Computer Science Engineering",
  },
  {
    year: "2024",
    place: { en: "UW–Madison", zh: "威斯康星大学麦迪逊分校" },
    detail: "MS / PhD, Computer Sciences",
  },
  { year: "2026", place: { en: "AMD GenAI", zh: "AMD GenAI" }, detail: "GPU kernels and LLM serving" },
];
