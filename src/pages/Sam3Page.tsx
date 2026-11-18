import { useMemo, useState } from "react";
import type { FC } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";

const BUCKET_URL = "https://openreal2sim-sam.s3.us-west-2.amazonaws.com/";
const RESULTS_URL = `${BUCKET_URL}auto_seg_1k/results.json`;
const GENERIC_URL = `${BUCKET_URL}auto_seg_1k/generic.json`;
const REFETCH_MS = 60_000;
const GALLERY_CAP = 200;
const THRESHOLD_MIN = 0.5;
const THRESHOLD_MAX = 1;
const THRESHOLD_STEP = 0.01;
const THRESHOLD_DEFAULT = 0.9;
const BIN_WIDTH = 0.05;
// Score cutoff of the n_extra entry shown as a stat tile.
const EXTRA_CUTOFF = "0.3";

type ViewMode = "frames" | "objects";
type IouMode = "plain" | "merged";
type NameMode = "names" | "nonames";

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

export const Sam3Page: FC = () => {
  const [threshold, setThreshold] = useState(THRESHOLD_DEFAULT);
  const [thresholdText, setThresholdText] = useState(
    THRESHOLD_DEFAULT.toFixed(2),
  );
  const [iouMode, setIouMode] = useState<IouMode>("plain");
  const [nameMode, setNameMode] = useState<NameMode>("names");
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

  const objects = useMemo(() => data?.objects ?? [], [data]);
  const noNames = nameMode === "nonames";
  const frameData = noNames ? generic : (data ?? null);
  const runs = useMemo(() => frameData?.runs ?? [], [frameData]);
  const hasFrameData =
    (data?.runs?.length ?? 0) > 0 || (generic?.runs?.length ?? 0) > 0;
  // The frame view is the default once the evaluation publishes the runs
  // array; before that only the object view has anything to show.
  const view: ViewMode = hasFrameData ? (viewChoice ?? "frames") : "objects";

  const headerData = view === "frames" && frameData ? frameData : data;
  const runsDone = headerData?.runs_done ?? 0;
  const runsTotal = headerData?.runs_total ?? 0;
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
              <span>updated {formatUpdated(headerData?.updated)}</span>
              {view === "frames" && noNames && frameData?.prompt && (
                <span className="font-mono break-all">
                  prompt "{frameData.prompt}"
                </span>
              )}
              <span className="font-mono">
                {runsDone} / {runsTotal} runs
                {typeof headerData?.runs_skipped === "number"
                  ? `, ${headerData.runs_skipped} skipped`
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

          {hasFrameData ? (
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
                  ariaLabel="prompt source"
                  value={nameMode}
                  onChange={setNameMode}
                  options={[
                    { value: "names", label: "names" },
                    { value: "nonames", label: "no names" },
                  ]}
                />
              )}
              {view === "frames" && (
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
          ) : (
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
            noNames && !generic ? (
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
