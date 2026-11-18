import { useMemo, useState } from "react";
import type { FC } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";

const BUCKET_URL = "https://openreal2sim-sam.s3.us-west-2.amazonaws.com/";
const RESULTS_URL = `${BUCKET_URL}auto_seg_1k/results.json`;
const GENERIC_URL = `${BUCKET_URL}auto_seg_1k/generic.json`;
const TOPK_URL = `${BUCKET_URL}auto_seg_1k/topk.json`;
const TOPK20_URL = `${BUCKET_URL}auto_seg_1k/topk20.json`;
const REFETCH_MS = 60_000;
const GALLERY_CAP = 200;
const THRESHOLD_MIN = 0.5;
const THRESHOLD_MAX = 1;
const THRESHOLD_STEP = 0.01;
const THRESHOLD_DEFAULT = 0.9;
const BIN_WIDTH = 0.05;
// Score cutoff of the n_extra entry shown as a stat tile.
const EXTRA_CUTOFF = "0.3";
// Values of k offered when topk.json does not say which ones it carries.
const K_FALLBACK = ["3", "5", "10"];
const K_DEFAULT = "5";
// Range of the k slider driven by topk20.json.
const K_MIN = 1;
const K_MAX_FALLBACK = 20;
const K_SLIDER_DEFAULT = 5;
// A ranked entry below this IoU counts as an extra mask, not as a hit.
const EXTRA_IOU = 0.5;
// Steps of the "k needed to cover all" table.
const K_TABLE_STEPS = [1, 2, 3, 5, 10, 20];
const OVERLAY_DIR_FALLBACK = "auto_seg_1k/topk20/";
const OVERLAY_EXT_FALLBACK = ".png";

type ViewMode = "frames" | "objects";
type IouMode = "plain" | "merged";
type NameMode = "names" | "nonames" | "topk";

interface Sam3Object {
  run: string;
  oid: string;
  name?: string;
  n_prop?: number;
  score_top?: number;
  iou_top?: number;
  iou_oracle?: number;
  iou_match?: number;
  iou_merge?: number;
  overlay?: string;
}

interface Sam3RunObject {
  oid?: string;
  name?: string;
  n_prop?: number;
  score_top?: number;
  iou_match?: number;
  iou_merge?: number;
  iou_top?: number;
  iou_oracle?: number;
}

interface Sam3Run {
  run?: string;
  n_objects?: number;
  iou_min?: number;
  iou_mean?: number;
  iou_min_merge?: number;
  overlay?: string;
  objects?: Sam3RunObject[];
  // Proposals that matched no ground-truth object, keyed by score cutoff.
  n_extra?: Record<string, number>;
}

// One proposal kept by the top-k run that matched no ground-truth object.
interface Sam3Extra {
  rank?: number;
  score?: number;
  area_frac?: number;
}

interface Sam3TopkFrame {
  run?: string;
  k?: number;
  covered?: boolean;
  n_gt?: number;
  n_extra?: number;
  iou_min?: number;
  extras?: Sam3Extra[];
  overlay?: string;
}

interface Sam3Topk {
  updated?: string;
  prompt?: string;
  ks?: number[];
  runs_total?: number;
  by_k?: Record<string, Sam3TopkFrame[]>;
}

// One entry of the ranked proposal list of a topk20.json run: gt is the
// ground-truth name it matched, or null when it matched nothing.
interface Sam3Ranked {
  rank?: number;
  score?: number;
  area_frac?: number;
  gt?: string | null;
  iou?: number | null;
}

interface Sam3Topk20Run {
  run?: string;
  n_gt?: number;
  gt_names?: string[];
  ranked?: Sam3Ranked[];
}

interface Sam3Topk20 {
  updated?: string;
  prompt?: string;
  k_max?: number;
  runs_total?: number;
  overlay_dir?: string;
  overlay_ext?: string;
  runs?: Sam3Topk20Run[];
}

interface Sam3Results {
  updated?: string;
  prompt?: string;
  runs_done?: number;
  runs_total?: number;
  runs_skipped?: number;
  objects?: Sam3Object[];
  runs?: Sam3Run[];
}

// 403 and 404 mean the evaluation has not written results.json yet, so they are
// not errors: return null and let the poll pick the file up once it lands.
async function getSam3Results(): Promise<Sam3Results | null> {
  const response = await fetch(RESULTS_URL, { cache: "no-store" });
  if (response.status === 403 || response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`Failed to fetch results.json (${response.status})`);
  }
  return (await response.json()) as Sam3Results;
}

async function getSam3Generic(): Promise<Sam3Results | null> {
  const response = await fetch(GENERIC_URL, { cache: "no-store" });
  if (response.status === 403 || response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`Failed to fetch generic.json (${response.status})`);
  }
  return (await response.json()) as Sam3Results;
}

async function getSam3Topk(): Promise<Sam3Topk | null> {
  const response = await fetch(TOPK_URL, { cache: "no-store" });
  if (response.status === 403 || response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`Failed to fetch topk.json (${response.status})`);
  }
  return (await response.json()) as Sam3Topk;
}

async function getSam3Topk20(): Promise<Sam3Topk20 | null> {
  const response = await fetch(TOPK20_URL, { cache: "no-store" });
  if (response.status === 403 || response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`Failed to fetch topk20.json (${response.status})`);
  }
  return (await response.json()) as Sam3Topk20;
}

function formatIou(value?: number): string {
  return typeof value === "number" ? value.toFixed(3) : "n/a";
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatUpdated(value?: string): string {
  if (!value) return "unknown";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

function clampThreshold(value: number): number {
  if (Number.isNaN(value)) return THRESHOLD_DEFAULT;
  return Math.min(THRESHOLD_MAX, Math.max(THRESHOLD_MIN, value));
}

function isNumber(value: unknown): value is number {
  return typeof value === "number" && !Number.isNaN(value);
}

function mean(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

// In merged mode the merge field drives the page, falling back to the plain
// field when the evaluation did not publish a merged value for an entry.
function runIou(run: Sam3Run, mode: IouMode): number | undefined {
  const value =
    mode === "merged" ? (run.iou_min_merge ?? run.iou_min) : run.iou_min;
  return isNumber(value) ? value : undefined;
}

function objectIou(object: Sam3RunObject, mode: IouMode): number | undefined {
  const value =
    mode === "merged"
      ? (object.iou_merge ?? object.iou_match)
      : object.iou_match;
  return isNumber(value) ? value : undefined;
}

// topk.json carries its own covered flag at a fixed IoU, but the slider has to
// drive the page, so covered is recomputed from iou_min whenever it is present.
function frameCovered(frame: Sam3TopkFrame, threshold: number): boolean {
  if (isNumber(frame.iou_min)) return frame.iou_min >= threshold;
  return frame.covered === true;
}

function frameExtras(frame: Sam3TopkFrame): number {
  if (isNumber(frame.n_extra)) return frame.n_extra;
  return frame.extras?.length ?? 0;
}

function formatExtraChip(extra: Sam3Extra): string {
  const rank = isNumber(extra.rank) ? `r${extra.rank}` : "r?";
  const score = isNumber(extra.score) ? `s${extra.score.toFixed(2)}` : "s?";
  const area = isNumber(extra.area_frac)
    ? `${(extra.area_frac * 100).toFixed(0)}%`
    : "?%";
  return `${rank} ${score} ${area}`;
}

// The k values to offer: whatever the file declares, plus whatever it actually
// carries, falling back to the three the evaluation runs.
function kOptions(topk: Sam3Topk): string[] {
  const keys = new Set<string>();
  for (const value of topk.ks ?? []) {
    if (isNumber(value)) keys.add(String(value));
  }
  for (const key of Object.keys(topk.by_k ?? {})) {
    keys.add(key);
  }
  const list = keys.size > 0 ? [...keys] : K_FALLBACK;
  return list.sort((a, b) => Number(a) - Number(b));
}

// Overlays of topk20.json live in one directory per k, named after the last
// segment of the run id.
function runBasename(run: string): string {
  const parts = run.split("/").filter((part) => part.length > 0);
  return parts.length > 0 ? parts[parts.length - 1] : run;
}

function overlayUrl(topk20: Sam3Topk20, run: string, k: number): string {
  const rawDir = topk20.overlay_dir ?? OVERLAY_DIR_FALLBACK;
  const dir = rawDir.endsWith("/") ? rawDir : `${rawDir}/`;
  const ext = topk20.overlay_ext ?? OVERLAY_EXT_FALLBACK;
  return `${BUCKET_URL}${dir}k${k}/${runBasename(run)}${ext}`;
}

function topk20KMax(topk20: Sam3Topk20): number {
  if (!isNumber(topk20.k_max) || topk20.k_max < K_MIN) return K_MAX_FALLBACK;
  return Math.floor(topk20.k_max);
}

// The rank of a ranked entry, falling back to its position in the list when
// the evaluation did not write the field.
function entryRank(entry: Sam3Ranked, index: number): number {
  return isNumber(entry.rank) ? entry.rank : index + 1;
}

function entryName(entry: Sam3Ranked): string | null {
  return typeof entry.gt === "string" && entry.gt.length > 0 ? entry.gt : null;
}

function entryIou(entry: Sam3Ranked): number | null {
  return isNumber(entry.iou) ? entry.iou : null;
}

interface Topk20Eval {
  // One entry per ground-truth object still not matched, so a scene with two
  // objects of the same name can be half covered.
  missed: string[];
  nExtra: number;
  covered: boolean;
}

// Coverage of one frame at a given k and IoU threshold, counted per name so
// duplicate ground-truth names need one matching mask each.
function evaluateTopk20Run(
  run: Sam3Topk20Run,
  k: number,
  threshold: number,
): Topk20Eval {
  const needed = new Map<string, number>();
  for (const name of run.gt_names ?? []) {
    if (typeof name !== "string") continue;
    needed.set(name, (needed.get(name) ?? 0) + 1);
  }

  const hits = new Map<string, number>();
  let nExtra = 0;
  (run.ranked ?? []).forEach((entry, index) => {
    if (entryRank(entry, index) > k) return;
    const name = entryName(entry);
    const iou = entryIou(entry);
    if (name === null || (iou !== null && iou < EXTRA_IOU)) nExtra += 1;
    if (name !== null && iou !== null && iou >= threshold) {
      hits.set(name, (hits.get(name) ?? 0) + 1);
    }
  });

  const missed: string[] = [];
  for (const [name, count] of needed) {
    const have = hits.get(name) ?? 0;
    for (let index = have; index < count; index += 1) missed.push(name);
  }

  return { missed, nExtra, covered: missed.length === 0 };
}

// The smallest k that covers every ground-truth object at the threshold, or
// null when no k up to k_max does. Computed from ranked so it follows the
// slider instead of the precomputed field of the file.
function topk20KNeeded(
  run: Sam3Topk20Run,
  kMax: number,
  threshold: number,
): number | null {
  for (let k = K_MIN; k <= kMax; k += 1) {
    if (evaluateTopk20Run(run, k, threshold).covered) return k;
  }
  return null;
}

interface RankedChip {
  label: string;
  extra: boolean;
}

// Chips read "r1 pen 0.96" for a match and "r2 extra 0.10" for a mask that
// matched no ground-truth object.
function rankedChips(run: Sam3Topk20Run, k: number): RankedChip[] {
  const chips: RankedChip[] = [];
  (run.ranked ?? []).forEach((entry, index) => {
    const rank = entryRank(entry, index);
    if (rank > k) return;
    const name = entryName(entry);
    const iou = entryIou(entry);
    if (name === null) {
      const score = isNumber(entry.score) ? entry.score.toFixed(2) : "n/a";
      chips.push({ label: `r${rank} extra ${score}`, extra: true });
      return;
    }
    const shown = iou === null ? "n/a" : iou.toFixed(2);
    chips.push({
      label: `r${rank} ${name} ${shown}`,
      extra: iou !== null && iou < EXTRA_IOU,
    });
  });
  return chips;
}

interface Bin {
  label: string;
  start: number;
  end: number;
  count: number;
}

function buildBins(values: number[]): Bin[] {
  const bins: Bin[] = [
    { label: "<0.5", start: Number.NEGATIVE_INFINITY, end: 0.5, count: 0 },
  ];
  for (let edge = 0.5; edge < 1 - 1e-9; edge += BIN_WIDTH) {
    const start = Math.round(edge * 100) / 100;
    const end = Math.round((edge + BIN_WIDTH) * 100) / 100;
    bins.push({ label: start.toFixed(2), start, end, count: 0 });
  }

  for (const value of values) {
    if (!isNumber(value)) continue;
    if (value < 0.5) {
      bins[0].count += 1;
      continue;
    }
    // The last bin is closed on the right so an IoU of exactly 1 lands in it.
    let index = Math.floor((value - 0.5) / BIN_WIDTH) + 1;
    if (index > bins.length - 1) index = bins.length - 1;
    bins[index].count += 1;
  }

  return bins;
}

const StatTile: FC<{ label: string; value: string; note?: string }> = ({
  label,
  value,
  note,
}) => (
  <div className="flex min-w-[9rem] flex-1 flex-col gap-1 rounded-lg border border-slate-300 bg-slate-100/50 px-3 py-2">
    <span className="text-xs tracking-wide text-gray-500 uppercase">
      {label}
    </span>
    <span className="font-mono text-xl font-semibold text-gray-800">
      {value}
    </span>
    {note && <span className="text-xs text-gray-500">{note}</span>}
  </div>
);

interface ToggleOption<T extends string> {
  value: T;
  label: string;
}

function Toggle<T extends string>({
  options,
  value,
  onChange,
  ariaLabel,
}: {
  options: ToggleOption<T>[];
  value: T;
  onChange: (next: T) => void;
  ariaLabel: string;
}) {
  return (
    <div
      className="inline-flex overflow-hidden rounded-md border border-slate-300 bg-white"
      role="group"
      aria-label={ariaLabel}
    >
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          aria-pressed={option.value === value}
          onClick={() => onChange(option.value)}
          className={
            option.value === value
              ? "px-3 py-1 text-sm font-medium text-white bg-slate-700"
              : "px-3 py-1 text-sm font-medium text-gray-600 hover:bg-slate-100"
          }
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

const Histogram: FC<{ title: string; bins: Bin[]; threshold: number }> = ({
  title,
  bins,
  threshold,
}) => {
  const maxBinCount = Math.max(1, ...bins.map((bin) => bin.count));
  return (
    <div className="flex flex-col gap-2 rounded-xl border border-slate-300 bg-slate-100/50 p-4">
      <h2 className="text-lg font-semibold text-gray-800">{title}</h2>
      <div className="flex h-40 items-end gap-1">
        {bins.map((bin) => (
          <div
            key={bin.label}
            className="flex h-full flex-1 flex-col justify-end gap-1"
            title={`${bin.label}: ${bin.count}`}
          >
            <span className="text-center font-mono text-xs text-gray-600">
              {bin.count}
            </span>
            <div
              className={
                bin.start < threshold
                  ? "w-full rounded-t bg-amber-500"
                  : "w-full rounded-t bg-slate-700"
              }
              style={{
                height: `${(bin.count / maxBinCount) * 100}%`,
                minHeight: bin.count > 0 ? "2px" : "0px",
              }}
            />
          </div>
        ))}
      </div>
      <div className="flex gap-1">
        {bins.map((bin) => (
          <span
            key={bin.label}
            className="flex-1 text-center font-mono text-[10px] text-gray-500"
          >
            {bin.label}
          </span>
        ))}
      </div>
    </div>
  );
};

const ObjectCard: FC<{ object: Sam3Object }> = ({ object }) => (
  <div className="flex flex-col gap-2 rounded-xl border border-slate-300 bg-slate-100/50 p-3 shadow-lg backdrop-blur-sm">
    {object.overlay ? (
      <img
        className="w-full rounded-lg border border-slate-300 bg-black"
        src={`${BUCKET_URL}${object.overlay}`}
        alt={`Top proposal overlay for ${object.run} object ${object.oid}`}
        loading="lazy"
      />
    ) : (
      <p className="rounded-lg border border-dashed border-slate-300 px-3 py-6 text-center text-sm text-gray-500">
        No overlay image
      </p>
    )}
    <div className="flex flex-col gap-0.5">
      <span className="font-mono text-sm break-all text-gray-700">
        {object.run} / {object.oid}
      </span>
      <span className="text-base font-semibold text-gray-800">
        {object.name ?? "unnamed"}
      </span>
      <span className="font-mono text-sm text-gray-600">
        iou_top {formatIou(object.iou_top)} | iou_oracle{" "}
        {formatIou(object.iou_oracle)}
      </span>
      <span className="font-mono text-sm text-gray-600">
        n_prop {object.n_prop ?? "n/a"} | score_top{" "}
        {formatIou(object.score_top)}
      </span>
    </div>
  </div>
);

const RunCard: FC<{ run: Sam3Run; threshold: number; iouMode: IouMode }> = ({
  run,
  threshold,
  iouMode,
}) => {
  const driving = runIou(run, iouMode);
  const objects = run.objects ?? [];
  return (
    <div className="flex flex-col gap-2 rounded-xl border border-slate-300 bg-slate-100/50 p-3 shadow-lg backdrop-blur-sm">
      {run.overlay ? (
        <img
          className="w-full rounded-lg border border-slate-300 bg-black"
          src={`${BUCKET_URL}${run.overlay}`}
          alt={`Frame 0 overlay for ${run.run ?? "run"}`}
          loading="lazy"
        />
      ) : (
        <p className="rounded-lg border border-dashed border-slate-300 px-3 py-6 text-center text-sm text-gray-500">
          No overlay image
        </p>
      )}
      <div className="flex flex-col gap-0.5">
        <span className="font-mono text-sm break-all text-gray-700">
          {run.run ?? "unknown run"}
        </span>
        <span className="font-mono text-sm text-gray-600">
          iou_min {formatIou(driving)} | iou_mean {formatIou(run.iou_mean)} |{" "}
          {run.n_objects ?? objects.length} objects
        </span>
      </div>
      {objects.length > 0 && (
        <div className="flex flex-col gap-0.5 border-t border-slate-300 pt-2">
          {objects.map((object, index) => {
            const objectDriving = objectIou(object, iouMode);
            const below =
              objectDriving !== undefined && objectDriving < threshold;
            return (
              <span
                key={object.oid ?? index}
                className={
                  below
                    ? "font-mono text-xs text-red-600"
                    : "font-mono text-xs text-gray-600"
                }
              >
                {object.name ?? "unnamed"} | iou_match{" "}
                {formatIou(object.iou_match)} | iou_merge{" "}
                {formatIou(object.iou_merge)} | score_top{" "}
                {formatIou(object.score_top)}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
};

const ObjectsView: FC<{
  objects: Sam3Object[];
  threshold: number;
  runsDone: number;
}> = ({ objects, threshold, runsDone }) => {
  const bins = useMemo(
    () => buildBins(objects.map((object) => object.iou_top ?? Number.NaN)),
    [objects],
  );

  const stats = useMemo(() => {
    const total = objects.length;
    const below = objects.filter((object) => (object.iou_top ?? 0) < threshold);
    const runsWithBelow = new Set(below.map((object) => object.run)).size;
    const rescued = below.filter(
      (object) => (object.iou_oracle ?? 0) >= threshold,
    ).length;
    const meanTop = mean(
      objects.map((object) => object.iou_top).filter(isNumber),
    );
    const meanOracle = mean(
      objects.map((object) => object.iou_oracle).filter(isNumber),
    );
    return { total, below, runsWithBelow, rescued, meanTop, meanOracle };
  }, [objects, threshold]);

  const gallery = useMemo(() => {
    const sorted = [...stats.below].sort(
      (a, b) => (a.iou_top ?? 0) - (b.iou_top ?? 0),
    );
    return sorted.slice(0, GALLERY_CAP);
  }, [stats.below]);

  return (
    <>
      <div className="flex flex-wrap gap-3">
        <StatTile label="objects" value={String(stats.total)} />
        <StatTile
          label="below threshold"
          value={String(stats.below.length)}
          note={
            stats.total > 0
              ? formatPercent(stats.below.length / stats.total)
              : "n/a"
          }
        />
        <StatTile
          label="runs with a miss"
          value={String(stats.runsWithBelow)}
          note={
            runsDone > 0 ? formatPercent(stats.runsWithBelow / runsDone) : "n/a"
          }
        />
        <StatTile
          label="mean iou_top"
          value={stats.meanTop === null ? "n/a" : stats.meanTop.toFixed(3)}
        />
        <StatTile
          label="mean iou_oracle"
          value={
            stats.meanOracle === null ? "n/a" : stats.meanOracle.toFixed(3)
          }
        />
        <StatTile
          label="oracle rescues"
          value={String(stats.rescued)}
          note={
            stats.below.length > 0
              ? `${formatPercent(stats.rescued / stats.below.length)} of misses`
              : "n/a"
          }
        />
      </div>

      <Histogram
        title="iou_top distribution"
        bins={bins}
        threshold={threshold}
      />

      <div className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold text-gray-800">
          Objects below {threshold.toFixed(2)}
        </h2>
        {stats.below.length === 0 ? (
          <p className="text-gray-600">No objects below the threshold yet.</p>
        ) : (
          <>
            <p className="text-sm text-gray-600">
              showing {gallery.length} of {stats.below.length}
            </p>
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              {gallery.map((object) => (
                <ObjectCard
                  key={`${object.run}/${object.oid}`}
                  object={object}
                />
              ))}
            </div>
          </>
        )}
      </div>
    </>
  );
};

const FramesView: FC<{
  runs: Sam3Run[];
  threshold: number;
  iouMode: IouMode;
  runsDone: number;
  showExtra: boolean;
}> = ({ runs, threshold, iouMode, runsDone, showExtra }) => {
  const bins = useMemo(
    () => buildBins(runs.map((run) => runIou(run, iouMode) ?? Number.NaN)),
    [runs, iouMode],
  );

  const stats = useMemo(() => {
    const framesDone = runsDone > 0 ? runsDone : runs.length;

    const missed = runs.filter((run) => {
      const value = runIou(run, iouMode);
      return value !== undefined && value < threshold;
    });

    let objectsTotal = 0;
    let objectsBelow = 0;
    for (const run of runs) {
      const objects = run.objects ?? [];
      objectsTotal +=
        objects.length > 0 ? objects.length : (run.n_objects ?? 0);
      for (const object of objects) {
        const value = objectIou(object, iouMode);
        if (value !== undefined && value < threshold) objectsBelow += 1;
      }
    }

    // A rescue is a frame that misses on the plain IoU but clears the
    // threshold once the proposals are merged.
    const rescued = runs.filter(
      (run) =>
        isNumber(run.iou_min) &&
        run.iou_min < threshold &&
        isNumber(run.iou_min_merge) &&
        run.iou_min_merge >= threshold,
    ).length;

    const meanIouMin = mean(
      runs.map((run) => runIou(run, iouMode)).filter(isNumber),
    );

    const meanExtra = mean(
      runs.map((run) => run.n_extra?.[EXTRA_CUTOFF]).filter(isNumber),
    );

    return {
      framesDone,
      missed,
      objectsTotal,
      objectsBelow,
      rescued,
      meanIouMin,
      meanExtra,
    };
  }, [runs, threshold, iouMode, runsDone]);

  const gallery = useMemo(() => {
    const sorted = [...stats.missed].sort(
      (a, b) => (runIou(a, iouMode) ?? 0) - (runIou(b, iouMode) ?? 0),
    );
    return sorted.slice(0, GALLERY_CAP);
  }, [stats.missed, iouMode]);

  const metricName = iouMode === "merged" ? "iou_min_merge" : "iou_min";

  return (
    <>
      <div className="flex flex-wrap gap-3">
        <StatTile label="frames done" value={String(stats.framesDone)} />
        <StatTile
          label="frames below threshold"
          value={String(stats.missed.length)}
          note={
            stats.framesDone > 0
              ? formatPercent(stats.missed.length / stats.framesDone)
              : "n/a"
          }
        />
        <StatTile
          label="objects below threshold"
          value={String(stats.objectsBelow)}
          note={
            stats.objectsTotal > 0
              ? `${formatPercent(stats.objectsBelow / stats.objectsTotal)} of ${stats.objectsTotal}`
              : "n/a"
          }
        />
        <StatTile
          label={`mean ${metricName}`}
          value={
            stats.meanIouMin === null ? "n/a" : stats.meanIouMin.toFixed(3)
          }
        />
        <StatTile
          label="rescued by merge"
          value={String(stats.rescued)}
          note="plain miss, merged pass"
        />
        {showExtra && (
          <StatTile
            label="extra masks per frame (mean)"
            value={
              stats.meanExtra === null ? "n/a" : stats.meanExtra.toFixed(2)
            }
            note={`n_extra at score ${EXTRA_CUTOFF}`}
          />
        )}
      </div>

      <Histogram
        title={`${metricName} distribution`}
        bins={bins}
        threshold={threshold}
      />

      <div className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold text-gray-800">
          Frames below {threshold.toFixed(2)}
        </h2>
        {stats.missed.length === 0 ? (
          <p className="text-gray-600">No frames below the threshold yet.</p>
        ) : (
          <>
            <p className="text-sm text-gray-600">
              showing {gallery.length} of {stats.missed.length}
            </p>
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              {gallery.map((run, index) => (
                <RunCard
                  key={run.run ?? `run-${index}`}
                  run={run}
                  threshold={threshold}
                  iouMode={iouMode}
                />
              ))}
            </div>
          </>
        )}
      </div>
    </>
  );
};

const TopkCard: FC<{ frame: Sam3TopkFrame; threshold: number }> = ({
  frame,
  threshold,
}) => {
  const covered = frameCovered(frame, threshold);
  const extras = frame.extras ?? [];
  return (
    <div className="flex flex-col gap-2 rounded-xl border border-slate-300 bg-slate-100/50 p-3 shadow-lg backdrop-blur-sm">
      {frame.overlay ? (
        <img
          className="w-full rounded-lg border border-slate-300 bg-black"
          src={`${BUCKET_URL}${frame.overlay}`}
          alt={`Top-k overlay for ${frame.run ?? "run"}`}
          loading="lazy"
        />
      ) : (
        <p className="rounded-lg border border-dashed border-slate-300 px-3 py-6 text-center text-sm text-gray-500">
          No overlay image
        </p>
      )}
      <div className="flex flex-col gap-1">
        <span className="font-mono text-sm break-all text-gray-700">
          {frame.run ?? "unknown run"}
        </span>
        <div className="flex flex-wrap items-center gap-2">
          <span
            className={
              covered
                ? "rounded-full bg-slate-700 px-2 py-0.5 text-xs font-medium text-white"
                : "rounded-full bg-amber-500 px-2 py-0.5 text-xs font-medium text-white"
            }
          >
            {covered ? "covered" : "not covered"}
          </span>
          <span className="font-mono text-sm text-gray-600">
            n_extra {frameExtras(frame)} | iou_min {formatIou(frame.iou_min)} |
            n_gt {frame.n_gt ?? "n/a"}
          </span>
        </div>
        {extras.length > 0 && (
          <div className="flex flex-wrap gap-1 border-t border-slate-300 pt-2">
            {extras.map((extra, index) => (
              <span
                key={`${extra.rank ?? "r"}-${index}`}
                className="rounded border border-slate-300 bg-white px-1.5 py-0.5 font-mono text-[11px] text-gray-600"
              >
                {formatExtraChip(extra)}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

const TopkView: FC<{ topk: Sam3Topk; threshold: number }> = ({
  topk,
  threshold,
}) => {
  const [kChoice, setKChoice] = useState<string | null>(null);
  const [onlyExtras, setOnlyExtras] = useState(true);

  const ks = useMemo(() => kOptions(topk), [topk]);
  const k =
    kChoice !== null && ks.includes(kChoice)
      ? kChoice
      : ks.includes(K_DEFAULT)
        ? K_DEFAULT
        : (ks[0] ?? K_DEFAULT);
  const frames = useMemo(() => topk.by_k?.[k] ?? [], [topk, k]);

  const stats = useMemo(() => {
    const total = frames.length;
    const covered = frames.filter((frame) => frameCovered(frame, threshold));
    const withExtras = frames.filter((frame) => frameExtras(frame) > 0).length;
    const meanExtras = mean(frames.map(frameExtras));
    return { total, covered: covered.length, withExtras, meanExtras };
  }, [frames, threshold]);

  // Most extras first so the frames that carry the junk masks come up top, and
  // the weakest coverage first inside a tie.
  const picked = useMemo(() => {
    const kept = onlyExtras
      ? frames.filter((frame) => frameExtras(frame) > 0)
      : frames;
    return [...kept].sort((a, b) => {
      const byExtras = frameExtras(b) - frameExtras(a);
      if (byExtras !== 0) return byExtras;
      return (a.iou_min ?? 0) - (b.iou_min ?? 0);
    });
  }, [frames, onlyExtras]);
  const gallery = picked.slice(0, GALLERY_CAP);

  return (
    <>
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-sm font-medium text-gray-700">top-k</span>
        <Toggle
          ariaLabel="top-k value"
          value={k}
          onChange={setKChoice}
          options={ks.map((option) => ({ value: option, label: option }))}
        />
        <label className="flex items-center gap-2 text-sm text-gray-700">
          <input
            className="h-4 w-4 accent-slate-700"
            type="checkbox"
            checked={onlyExtras}
            onChange={(event) => setOnlyExtras(event.target.checked)}
          />
          only frames with extras
        </label>
      </div>

      <div className="flex flex-wrap gap-3">
        <StatTile label="frames" value={String(stats.total)} />
        <StatTile
          label="frames covered"
          value={String(stats.covered)}
          note={
            stats.total > 0 ? formatPercent(stats.covered / stats.total) : "n/a"
          }
        />
        <StatTile
          label="frames with extras"
          value={String(stats.withExtras)}
          note={
            stats.total > 0
              ? formatPercent(stats.withExtras / stats.total)
              : "n/a"
          }
        />
        <StatTile
          label="mean extras per frame"
          value={
            stats.meanExtras === null ? "n/a" : stats.meanExtras.toFixed(2)
          }
          note={`k ${k}`}
        />
      </div>

      <div className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold text-gray-800">
          {onlyExtras ? "Frames with extras" : "All frames"} at k {k}
        </h2>
        {picked.length === 0 ? (
          <p className="text-gray-600">No frames to show at k {k}.</p>
        ) : (
          <>
            <p className="text-sm text-gray-600">
              showing {gallery.length} of {picked.length}
            </p>
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              {gallery.map((frame, index) => (
                <TopkCard
                  key={frame.run ?? `frame-${index}`}
                  frame={frame}
                  threshold={threshold}
                />
              ))}
            </div>
          </>
        )}
      </div>
    </>
  );
};

type CoverFilter = "missed" | "all" | "covered";

const COVER_FILTER_TITLE: Record<CoverFilter, string> = {
  missed: "Frames missing something",
  all: "All frames",
  covered: "Fully covered frames",
};

interface Topk20Row extends Topk20Eval {
  run: Sam3Topk20Run;
}

interface KNeededRow {
  label: string;
  count: number;
}

const Topk20Card: FC<{
  row: Topk20Row;
  k: number;
  overlay: string;
}> = ({ row, k, overlay }) => {
  const chips = rankedChips(row.run, k);
  const name = row.run.run ?? "unknown run";
  const nGt = row.run.n_gt ?? row.run.gt_names?.length;
  return (
    <div className="flex flex-col gap-2 rounded-xl border border-slate-300 bg-slate-100/50 p-3 shadow-lg backdrop-blur-sm">
      <img
        className="w-full rounded-lg border border-slate-300 bg-black"
        src={overlay}
        alt={`Top ${k} overlay for ${name}`}
        loading="lazy"
      />
      <div className="flex flex-col gap-1">
        <span className="font-mono text-sm break-all text-gray-700">
          {name}
        </span>
        <div className="flex flex-wrap items-center gap-2">
          <span
            className={
              row.covered
                ? "rounded-full bg-slate-700 px-2 py-0.5 text-xs font-medium text-white"
                : "rounded-full bg-amber-500 px-2 py-0.5 text-xs font-medium text-white"
            }
          >
            {row.covered ? "covered" : "missed"}
          </span>
          <span className="font-mono text-sm text-gray-600">
            n_extra {row.nExtra} | n_gt {nGt ?? "n/a"}
          </span>
        </div>
        {row.missed.length > 0 && (
          <span className="font-mono text-sm font-semibold text-red-600">
            missed {row.missed.join(", ")}
          </span>
        )}
        {chips.length > 0 && (
          <div className="flex flex-wrap gap-1 border-t border-slate-300 pt-2">
            {chips.map((chip, index) => (
              <span
                key={`${chip.label}-${index}`}
                className={
                  chip.extra
                    ? "rounded border border-amber-300 bg-amber-50 px-1.5 py-0.5 font-mono text-[11px] text-amber-700"
                    : "rounded border border-slate-300 bg-white px-1.5 py-0.5 font-mono text-[11px] text-gray-600"
                }
              >
                {chip.label}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

const KNeededTable: FC<{
  rows: KNeededRow[];
  total: number;
  threshold: number;
}> = ({ rows, total, threshold }) => (
  <div className="flex min-w-[14rem] flex-col gap-1 rounded-lg border border-slate-300 bg-slate-100/50 px-3 py-2">
    <span className="text-xs tracking-wide text-gray-500 uppercase">
      k needed to cover all at {threshold.toFixed(2)}
    </span>
    <table className="font-mono text-sm text-gray-700">
      <tbody>
        {rows.map((row) => (
          <tr key={row.label}>
            <td className="pr-4 text-gray-500">{row.label}</td>
            <td className="pr-3 text-right font-semibold">{row.count}</td>
            <td className="text-right text-gray-500">
              {total > 0 ? formatPercent(row.count / total) : "n/a"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

const Topk20View: FC<{ topk20: Sam3Topk20; threshold: number }> = ({
  topk20,
  threshold,
}) => {
  const [kChoice, setKChoice] = useState(K_SLIDER_DEFAULT);
  const [coverFilter, setCoverFilter] = useState<CoverFilter>("missed");
  const [onlyExtras, setOnlyExtras] = useState(false);

  const kMax = topk20KMax(topk20);
  const k = Math.min(Math.max(kChoice, K_MIN), kMax);
  const runs = useMemo(() => topk20.runs ?? [], [topk20]);

  const rows = useMemo(
    () => runs.map((run) => ({ run, ...evaluateTopk20Run(run, k, threshold) })),
    [runs, k, threshold],
  );

  const stats = useMemo(() => {
    const total = rows.length;
    const covered = rows.filter((row) => row.covered).length;
    const meanExtras = mean(rows.map((row) => row.nExtra));
    return { total, covered, missed: total - covered, meanExtras };
  }, [rows]);

  // Smallest covering k per frame, recomputed from ranked so the table follows
  // the IoU slider.
  const kTable = useMemo(() => {
    const needed = runs.map((run) => topk20KNeeded(run, kMax, threshold));
    const steps = K_TABLE_STEPS.filter((step) => step <= kMax);
    if (steps.length === 0 || steps[steps.length - 1] !== kMax)
      steps.push(kMax);
    const table: KNeededRow[] = steps.map((step) => ({
      label: `k <= ${step}`,
      count: needed.filter((value) => value !== null && value <= step).length,
    }));
    table.push({
      label: "never",
      count: needed.filter((value) => value === null).length,
    });
    return table;
  }, [runs, kMax, threshold]);

  // Worst frames first: most missing objects, then most extra masks.
  const picked = useMemo(() => {
    const kept = rows.filter((row) => {
      if (coverFilter === "missed" && row.covered) return false;
      if (coverFilter === "covered" && !row.covered) return false;
      if (onlyExtras && row.nExtra === 0) return false;
      return true;
    });
    return [...kept].sort((a, b) => {
      const byMissed = b.missed.length - a.missed.length;
      if (byMissed !== 0) return byMissed;
      return b.nExtra - a.nExtra;
    });
  }, [rows, coverFilter, onlyExtras]);
  const gallery = picked.slice(0, GALLERY_CAP);

  return (
    <>
      <div className="flex flex-col gap-3 rounded-xl border border-slate-300 bg-slate-100/50 p-4">
        <div className="flex items-center gap-4">
          <span className="text-sm font-medium text-gray-700">top-k</span>
          <span className="font-mono text-4xl font-bold text-gray-800">
            {k}
          </span>
          <span className="text-sm text-gray-500">of {kMax}</span>
        </div>
        <input
          className="h-2 w-full cursor-pointer accent-slate-700"
          type="range"
          min={K_MIN}
          max={kMax}
          step={1}
          value={k}
          onChange={(event) => setKChoice(Number(event.target.value))}
          aria-label="top-k value"
        />
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <Toggle
          ariaLabel="coverage filter"
          value={coverFilter}
          onChange={setCoverFilter}
          options={[
            { value: "missed", label: "missed something" },
            { value: "all", label: "all" },
            { value: "covered", label: "fully covered" },
          ]}
        />
        <label className="flex items-center gap-2 text-sm text-gray-700">
          <input
            className="h-4 w-4 accent-slate-700"
            type="checkbox"
            checked={onlyExtras}
            onChange={(event) => setOnlyExtras(event.target.checked)}
          />
          only frames with extras
        </label>
      </div>

      <div className="flex flex-wrap gap-3">
        <StatTile label="frames" value={String(stats.total)} />
        <StatTile
          label="frames covered"
          value={String(stats.covered)}
          note={
            stats.total > 0 ? formatPercent(stats.covered / stats.total) : "n/a"
          }
        />
        <StatTile
          label="frames missing something"
          value={String(stats.missed)}
          note={
            stats.total > 0 ? formatPercent(stats.missed / stats.total) : "n/a"
          }
        />
        <StatTile
          label="mean extras per frame"
          value={
            stats.meanExtras === null ? "n/a" : stats.meanExtras.toFixed(2)
          }
          note={`k ${k}`}
        />
        <KNeededTable rows={kTable} total={stats.total} threshold={threshold} />
      </div>

      <div className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold text-gray-800">
          {COVER_FILTER_TITLE[coverFilter]} at k {k}
          {onlyExtras ? ", with extras" : ""}
        </h2>
        {picked.length === 0 ? (
          <p className="text-gray-600">No frames to show at k {k}.</p>
        ) : (
          <>
            <p className="text-sm text-gray-600">
              showing {gallery.length} of {picked.length}
            </p>
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              {gallery.map((row, index) => (
                <Topk20Card
                  key={row.run.run ?? `frame-${index}`}
                  row={row}
                  k={k}
                  overlay={overlayUrl(topk20, row.run.run ?? "", k)}
                />
              ))}
            </div>
          </>
        )}
      </div>
    </>
  );
};

export const Sam3Page: FC = () => {
  const [threshold, setThreshold] = useState(THRESHOLD_DEFAULT);
  const [thresholdText, setThresholdText] = useState(
    THRESHOLD_DEFAULT.toFixed(2),
  );
  const [iouMode, setIouMode] = useState<IouMode>("plain");
  const [modeChoice, setModeChoice] = useState<NameMode | null>(null);
  const [viewChoice, setViewChoice] = useState<ViewMode | null>(null);

  const { data, isPending, error } = useQuery({
    queryKey: ["sam3Results"],
    queryFn: getSam3Results,
    refetchInterval: REFETCH_MS,
    retry: false,
  });

  // The no-names run is a separate file written by the generic-prompt
  // evaluation; a missing file is not an error here either.
  const genericQuery = useQuery({
    queryKey: ["sam3Generic"],
    queryFn: getSam3Generic,
    refetchInterval: REFETCH_MS,
    retry: false,
  });
  const generic = genericQuery.data ?? null;

  // The top-k run writes its own file; a missing file is not an error either.
  const topkQuery = useQuery({
    queryKey: ["sam3Topk"],
    queryFn: getSam3Topk,
    refetchInterval: REFETCH_MS,
    retry: false,
  });
  const topk = topkQuery.data ?? null;

  // topk20.json is the file behind the k slider; while it is missing the page
  // falls back to the fixed k values of topk.json.
  const topk20Query = useQuery({
    queryKey: ["sam3Topk20"],
    queryFn: getSam3Topk20,
    refetchInterval: REFETCH_MS,
    retry: false,
  });
  const topk20 = topk20Query.data ?? null;
  const topk20Count = topk20?.runs?.length ?? 0;

  const objects = useMemo(() => data?.objects ?? [], [data]);

  // done is the longest per-k list, so the progress bar means the same thing
  // in top-k mode as it does for the run files.
  const topkCounts = useMemo(() => {
    const lists = Object.values(topk?.by_k ?? {});
    return {
      any: lists.reduce((count, list) => count + (list?.length ?? 0), 0),
      done: lists.reduce(
        (count, list) => Math.max(count, list?.length ?? 0),
        0,
      ),
    };
  }, [topk]);

  const hasTopkData = topkCounts.any > 0 || topk20Count > 0;
  const hasRunData =
    (data?.runs?.length ?? 0) > 0 || (generic?.runs?.length ?? 0) > 0;
  const hasFrameData = hasRunData || hasTopkData;
  // Top-k is the default data mode when it is the only frame-level file on the
  // bucket, otherwise the named run leads as before.
  const nameMode: NameMode =
    modeChoice ?? (hasRunData || !hasTopkData ? "names" : "topk");
  const noNames = nameMode === "nonames";
  const topkMode = nameMode === "topk";
  const frameData = noNames ? generic : (data ?? null);
  const runs = useMemo(() => frameData?.runs ?? [], [frameData]);
  // The frame view is the default once the evaluation publishes frame-level
  // data; before that the object view leads, but either stays reachable.
  const view: ViewMode = viewChoice ?? (hasFrameData ? "frames" : "objects");

  const headerData =
    view === "frames" && !topkMode && frameData ? frameData : data;
  const headerUpdated = topkMode
    ? (topk20?.updated ?? topk?.updated)
    : headerData?.updated;
  const headerPrompt = topkMode
    ? (topk20?.prompt ?? topk?.prompt)
    : noNames
      ? frameData?.prompt
      : undefined;
  const runsDone = topkMode
    ? Math.max(topk20Count, topkCounts.done)
    : (headerData?.runs_done ?? 0);
  const runsTotal =
    (topkMode
      ? (topk20?.runs_total ?? topk?.runs_total)
      : headerData?.runs_total) ?? 0;
  const runsSkipped = topkMode ? undefined : headerData?.runs_skipped;
  const progress = runsTotal > 0 ? Math.min(1, runsDone / runsTotal) : 0;

  const applyThreshold = (raw: string) => {
    setThresholdText(raw);
    const parsed = Number(raw);
    if (raw.trim() !== "" && !Number.isNaN(parsed)) {
      setThreshold(clampThreshold(parsed));
    }
  };

  const onSlider = (raw: string) => {
    const next = clampThreshold(Number(raw));
    setThreshold(next);
    setThresholdText(next.toFixed(2));
  };

  return (
    <div className="min-h-screen p-8">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-3xl font-bold tracking-tight text-gray-800">
          SAM 3 evaluation (temporary)
        </h1>
        <Button variant="outline" asChild>
          <Link to="/">← Back</Link>
        </Button>
      </div>

      {isPending && <div className="text-gray-600">Loading...</div>}

      {!isPending && error && (
        <div className="text-red-600">
          Failed to load results.json: {error.message}
        </div>
      )}

      {!isPending && !error && !data && (
        <div className="rounded-xl border border-slate-300 bg-slate-100/50 p-6 text-gray-600">
          No results yet. The evaluation has not written results.json to the
          bucket. This page retries every minute.
        </div>
      )}

      {!isPending && !error && data && (
        <div className="flex flex-col gap-6">
          <div className="flex flex-col gap-2">
            <div className="flex flex-wrap items-baseline justify-between gap-2 text-sm text-gray-600">
              <span>updated {formatUpdated(headerUpdated)}</span>
              {view === "frames" && headerPrompt && (
                <span className="font-mono break-all">
                  prompt "{headerPrompt}"
                </span>
              )}
              <span className="font-mono">
                {runsDone} / {runsTotal} runs
                {typeof runsSkipped === "number"
                  ? `, ${runsSkipped} skipped`
                  : ""}
              </span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-300">
              <div
                className="h-full rounded-full bg-slate-700"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <Toggle
              ariaLabel="view level"
              value={view}
              onChange={setViewChoice}
              options={[
                { value: "frames", label: "frames" },
                { value: "objects", label: "objects" },
              ]}
            />
            {view === "frames" && (
              <Toggle
                ariaLabel="data mode"
                value={nameMode}
                onChange={setModeChoice}
                options={[
                  { value: "names", label: "names" },
                  { value: "nonames", label: "no names" },
                  { value: "topk", label: "top-k" },
                ]}
              />
            )}
            {view === "frames" && !topkMode && (
              <Toggle
                ariaLabel="iou source"
                value={iouMode}
                onChange={setIouMode}
                options={[
                  { value: "plain", label: "plain" },
                  { value: "merged", label: "merged" },
                ]}
              />
            )}
          </div>

          {!hasRunData && !topkMode && (
            <p className="text-sm text-gray-600">
              frame-level data not published yet
            </p>
          )}

          <div className="flex flex-col gap-3 rounded-xl border border-slate-300 bg-slate-100/50 p-4">
            <div className="flex items-center gap-4">
              <span className="text-sm font-medium text-gray-700">
                IoU threshold
              </span>
              <span className="font-mono text-4xl font-bold text-gray-800">
                {threshold.toFixed(2)}
              </span>
              <input
                className="h-9 w-24 rounded-md border border-slate-300 bg-white px-2 font-mono text-sm text-gray-800"
                type="number"
                min={THRESHOLD_MIN}
                max={THRESHOLD_MAX}
                step={THRESHOLD_STEP}
                value={thresholdText}
                onChange={(event) => applyThreshold(event.target.value)}
                aria-label="IoU threshold value"
              />
            </div>
            <input
              className="h-2 w-full cursor-pointer accent-slate-700"
              type="range"
              min={THRESHOLD_MIN}
              max={THRESHOLD_MAX}
              step={THRESHOLD_STEP}
              value={threshold}
              onChange={(event) => onSlider(event.target.value)}
              aria-label="IoU threshold"
            />
          </div>

          {view === "frames" ? (
            topkMode ? (
              topk20 ? (
                <Topk20View topk20={topk20} threshold={threshold} />
              ) : topk ? (
                <>
                  <p className="text-sm text-gray-600">
                    slider data not published yet
                  </p>
                  <TopkView topk={topk} threshold={threshold} />
                </>
              ) : (
                <p className="text-sm text-gray-600">
                  top-k data not published yet
                </p>
              )
            ) : noNames && !generic ? (
              <p className="text-sm text-gray-600">
                no-names data not published yet
              </p>
            ) : (
              <FramesView
                runs={runs}
                threshold={threshold}
                iouMode={iouMode}
                runsDone={runsDone}
                showExtra={noNames}
              />
            )
          ) : (
            <ObjectsView
              objects={objects}
              threshold={threshold}
              runsDone={runsDone}
            />
          )}
        </div>
      )}
    </div>
  );
};
