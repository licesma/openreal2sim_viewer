import { useMemo, useState } from "react";
import type { FC } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";

const BUCKET_URL = "https://openreal2sim-sam.s3.us-west-2.amazonaws.com/";
const RESULTS_URL = `${BUCKET_URL}auto_seg_1k/results.json`;
const REFETCH_MS = 60_000;
const GALLERY_CAP = 200;
const THRESHOLD_MIN = 0.5;
const THRESHOLD_MAX = 1;
const THRESHOLD_STEP = 0.01;
const THRESHOLD_DEFAULT = 0.9;
const BIN_WIDTH = 0.05;

interface Sam3Object {
  run: string;
  oid: string;
  name?: string;
  n_prop?: number;
  score_top?: number;
  iou_top?: number;
  iou_oracle?: number;
  overlay?: string;
}

interface Sam3Results {
  updated?: string;
  runs_done?: number;
  runs_total?: number;
  runs_skipped?: number;
  objects?: Sam3Object[];
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

interface Bin {
  label: string;
  start: number;
  end: number;
  count: number;
}

function buildBins(objects: Sam3Object[]): Bin[] {
  const bins: Bin[] = [
    { label: "<0.5", start: Number.NEGATIVE_INFINITY, end: 0.5, count: 0 },
  ];
  for (let edge = 0.5; edge < 1 - 1e-9; edge += BIN_WIDTH) {
    const start = Math.round(edge * 100) / 100;
    const end = Math.round((edge + BIN_WIDTH) * 100) / 100;
    bins.push({ label: start.toFixed(2), start, end, count: 0 });
  }

  for (const object of objects) {
    const iou = object.iou_top;
    if (typeof iou !== "number") continue;
    if (iou < 0.5) {
      bins[0].count += 1;
      continue;
    }
    // The last bin is closed on the right so an IoU of exactly 1 lands in it.
    let index = Math.floor((iou - 0.5) / BIN_WIDTH) + 1;
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

export const Sam3Page: FC = () => {
  const [threshold, setThreshold] = useState(THRESHOLD_DEFAULT);
  const [thresholdText, setThresholdText] = useState(
    THRESHOLD_DEFAULT.toFixed(2),
  );

  const { data, isPending, error } = useQuery({
    queryKey: ["sam3Results"],
    queryFn: getSam3Results,
    refetchInterval: REFETCH_MS,
    retry: false,
  });

  const objects = useMemo(() => data?.objects ?? [], [data]);
  const bins = useMemo(() => buildBins(objects), [objects]);
  const maxBinCount = Math.max(1, ...bins.map((bin) => bin.count));

  const stats = useMemo(() => {
    const total = objects.length;
    const below = objects.filter(
      (object) => (object.iou_top ?? 0) < threshold,
    );
    const runsWithBelow = new Set(below.map((object) => object.run)).size;
    const rescued = below.filter(
      (object) => (object.iou_oracle ?? 0) >= threshold,
    ).length;

    const mean = (values: number[]) =>
      values.length > 0
        ? values.reduce((sum, value) => sum + value, 0) / values.length
        : null;
    const meanTop = mean(
      objects
        .map((object) => object.iou_top)
        .filter((value): value is number => typeof value === "number"),
    );
    const meanOracle = mean(
      objects
        .map((object) => object.iou_oracle)
        .filter((value): value is number => typeof value === "number"),
    );

    return { total, below, runsWithBelow, rescued, meanTop, meanOracle };
  }, [objects, threshold]);

  const gallery = useMemo(() => {
    const sorted = [...stats.below].sort(
      (a, b) => (a.iou_top ?? 0) - (b.iou_top ?? 0),
    );
    return sorted.slice(0, GALLERY_CAP);
  }, [stats.below]);

  const runsDone = data?.runs_done ?? 0;
  const runsTotal = data?.runs_total ?? 0;
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
              <span>updated {formatUpdated(data.updated)}</span>
              <span className="font-mono">
                {runsDone} / {runsTotal} runs
                {typeof data.runs_skipped === "number"
                  ? `, ${data.runs_skipped} skipped`
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
                runsDone > 0
                  ? formatPercent(stats.runsWithBelow / runsDone)
                  : "n/a"
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

          <div className="flex flex-col gap-2 rounded-xl border border-slate-300 bg-slate-100/50 p-4">
            <h2 className="text-lg font-semibold text-gray-800">
              iou_top distribution
            </h2>
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

          <div className="flex flex-col gap-3">
            <h2 className="text-lg font-semibold text-gray-800">
              Objects below {threshold.toFixed(2)}
            </h2>
            {stats.below.length === 0 ? (
              <p className="text-gray-600">
                No objects below the threshold yet.
              </p>
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
        </div>
      )}
    </div>
  );
};
