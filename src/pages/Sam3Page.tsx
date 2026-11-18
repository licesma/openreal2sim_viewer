import { useMemo, useState } from "react";
import type { FC } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";

const BUCKET_URL = "https://openreal2sim-sam.s3.us-west-2.amazonaws.com/";
const DINO_URL = `${BUCKET_URL}auto_seg_1k/dino.json`;
const REFETCH_MS = 60_000;
const GALLERY_CAP = 200;
const THRESHOLD_MIN = 0.5;
const THRESHOLD_MAX = 1;
const THRESHOLD_STEP = 0.01;
const THRESHOLD_DEFAULT = 0.9;
// How many names the missed-object breakdown lists.
const MISSED_NAME_CAP = 10;

// Shape of dino.json: the Grounding DINO plus SAM2 run, one entry per frame
// with the matched IoU per ground-truth object. Every field is optional,
// the file is written while the evaluation runs.
interface DinoObject {
  oid?: string;
  name?: string;
  iou_match?: number;
  score_match?: number;
}

interface DinoRun {
  run?: string;
  n_objects?: number;
  iou_min?: number;
  iou_mean?: number;
  // Masks that matched no ground-truth object.
  n_extra?: number;
  covered?: number | boolean;
  overlay?: string;
  objects?: DinoObject[];
}

interface DinoSummary {
  covered?: number;
  mean_extra?: number;
  perfect?: number;
  cov_iou?: number;
  extra_iou?: number;
}

interface Dino {
  updated?: string;
  method?: string;
  prompt?: string;
  cutoff?: string | number;
  nms?: string;
  runs_done?: number;
  runs_total?: number;
  summary?: DinoSummary;
  runs?: DinoRun[];
}

// 403 and 404 mean the evaluation has not written dino.json to the bucket yet,
// so they are not errors: return null and let the poll pick the file up once
// it lands.
async function getDino(): Promise<Dino | null> {
  const response = await fetch(DINO_URL, { cache: "no-store" });
  if (response.status === 403 || response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`Failed to fetch dino.json (${response.status})`);
  }
  return (await response.json()) as Dino;
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

// A frame is covered when its worst matched object clears the slider, so the
// page follows the threshold instead of the covered flag of the file.
function dinoCovered(run: DinoRun, threshold: number): boolean {
  if (isNumber(run.iou_min)) return run.iou_min >= threshold;
  const known = (run.objects ?? [])
    .map((object) => object.iou_match)
    .filter(isNumber);
  if (known.length > 0) return Math.min(...known) >= threshold;
  return run.covered === true || run.covered === 1;
}

function dinoExtras(run: DinoRun): number {
  return isNumber(run.n_extra) ? run.n_extra : 0;
}

// Names of the ground-truth objects this frame missed. An object without a
// numeric iou_match carries no evidence either way, so it is left out.
function dinoMissedNames(run: DinoRun, threshold: number): string[] {
  const missed: string[] = [];
  for (const object of run.objects ?? []) {
    if (!isNumber(object.iou_match)) continue;
    if (object.iou_match < threshold) missed.push(object.name ?? "unnamed");
  }
  return missed;
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

type CoverFilter = "missed" | "all" | "covered";

const COVER_FILTER_TITLE: Record<CoverFilter, string> = {
  missed: "Frames missing something",
  all: "All frames",
  covered: "Fully covered frames",
};

interface DinoRow {
  run: DinoRun;
  missed: string[];
  nExtra: number;
  covered: boolean;
}

interface NameCountRow {
  name: string;
  count: number;
}

// How often each ground-truth object is the one Grounding DINO plus SAM2 fails
// to cover, biggest first: the breakdown used to spot a pattern.
function missedNameCounts(rows: DinoRow[], cap: number): NameCountRow[] {
  const counts = new Map<string, number>();
  for (const row of rows) {
    for (const name of row.missed) {
      counts.set(name, (counts.get(name) ?? 0) + 1);
    }
  }
  const list = [...counts].map(([name, count]) => ({ name, count }));
  list.sort((a, b) => {
    const byCount = b.count - a.count;
    if (byCount !== 0) return byCount;
    return a.name.localeCompare(b.name);
  });
  return list.slice(0, cap);
}

const MissedNamesTable: FC<{
  rows: NameCountRow[];
  total: number;
  threshold: number;
}> = ({ rows, total, threshold }) => (
  <div className="flex min-w-[14rem] flex-col gap-1 rounded-lg border border-slate-300 bg-slate-100/50 px-3 py-2">
    <span className="text-xs tracking-wide text-gray-500 uppercase">
      missed objects by name at {threshold.toFixed(2)}
    </span>
    {rows.length === 0 ? (
      <span className="font-mono text-sm text-gray-600">none</span>
    ) : (
      <table className="font-mono text-sm text-gray-700">
        <tbody>
          {rows.map((row) => (
            <tr key={row.name}>
              <td className="pr-4 text-gray-500">{row.name}</td>
              <td className="pr-3 text-right font-semibold">{row.count}</td>
              <td className="text-right text-gray-500">
                {total > 0 ? formatPercent(row.count / total) : "n/a"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    )}
  </div>
);

const DinoCard: FC<{ row: DinoRow; threshold: number }> = ({
  row,
  threshold,
}) => {
  const run = row.run;
  const name = run.run ?? "unknown run";
  const objects = run.objects ?? [];
  return (
    <div className="flex flex-col gap-2 rounded-xl border border-slate-300 bg-slate-100/50 p-3 shadow-lg backdrop-blur-sm">
      {run.overlay ? (
        <img
          className="w-full rounded-lg border border-slate-300 bg-black"
          src={`${BUCKET_URL}${run.overlay}`}
          alt={`Grounding DINO overlay for ${name}`}
          loading="lazy"
        />
      ) : (
        <p className="rounded-lg border border-dashed border-slate-300 px-3 py-6 text-center text-sm text-gray-500">
          No overlay image
        </p>
      )}
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
            iou_min {formatIou(run.iou_min)} | iou_mean{" "}
            {formatIou(run.iou_mean)} | n_extra {row.nExtra} |{" "}
            {run.n_objects ?? objects.length} objects
          </span>
        </div>
        {row.missed.length > 0 && (
          <span className="font-mono text-sm font-semibold text-red-600">
            missed {row.missed.join(", ")}
          </span>
        )}
        {objects.length > 0 && (
          <div className="flex flex-col gap-0.5 border-t border-slate-300 pt-2">
            {objects.map((object, index) => {
              const below =
                isNumber(object.iou_match) && object.iou_match < threshold;
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
                  {formatIou(object.iou_match)} | score_match{" "}
                  {formatIou(object.score_match)}
                </span>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

const DinoView: FC<{ dino: Dino; threshold: number }> = ({
  dino,
  threshold,
}) => {
  const [coverFilter, setCoverFilter] = useState<CoverFilter>("missed");
  const [onlyExtras, setOnlyExtras] = useState(false);

  const runs = useMemo(() => dino.runs ?? [], [dino]);

  const rows = useMemo<DinoRow[]>(
    () =>
      runs.map((run) => ({
        run,
        missed: dinoMissedNames(run, threshold),
        nExtra: dinoExtras(run),
        covered: dinoCovered(run, threshold),
      })),
    [runs, threshold],
  );

  const stats = useMemo(() => {
    const total = rows.length;
    const covered = rows.filter((row) => row.covered).length;
    const perfect = rows.filter(
      (row) => row.covered && row.nExtra === 0,
    ).length;
    const meanExtras = mean(rows.map((row) => row.nExtra));
    const missedObjects = rows.reduce(
      (count, row) => count + row.missed.length,
      0,
    );
    return {
      total,
      covered,
      missed: total - covered,
      perfect,
      meanExtras,
      missedObjects,
    };
  }, [rows]);

  const missedNames = useMemo(
    () => missedNameCounts(rows, MISSED_NAME_CAP),
    [rows],
  );

  // Weakest coverage first, so the frames DINO misses worst come up top.
  const picked = useMemo(() => {
    const kept = rows.filter((row) => {
      if (coverFilter === "missed" && row.covered) return false;
      if (coverFilter === "covered" && !row.covered) return false;
      if (onlyExtras && row.nExtra === 0) return false;
      return true;
    });
    return [...kept].sort(
      (a, b) => (a.run.iou_min ?? 0) - (b.run.iou_min ?? 0),
    );
  }, [rows, coverFilter, onlyExtras]);
  const gallery = picked.slice(0, GALLERY_CAP);

  const cutoff =
    dino.cutoff === undefined || dino.cutoff === null
      ? "n/a"
      : String(dino.cutoff);

  return (
    <>
      <div className="flex flex-wrap items-center gap-x-6 gap-y-1 rounded-xl border border-slate-300 bg-slate-100/50 px-4 py-3 font-mono text-sm text-gray-700">
        <span className="break-all">{dino.method ?? "method n/a"}</span>
        <span className="break-all">prompt "{dino.prompt ?? "n/a"}"</span>
        <span>cutoff {cutoff}</span>
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
        />
        <StatTile
          label="perfect frames"
          value={String(stats.perfect)}
          note={
            stats.total > 0
              ? `${formatPercent(stats.perfect / stats.total)}, covered and no extras`
              : "covered and no extras"
          }
        />
        <MissedNamesTable
          rows={missedNames}
          total={stats.missedObjects}
          threshold={threshold}
        />
      </div>

      <div className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold text-gray-800">
          {COVER_FILTER_TITLE[coverFilter]} at {threshold.toFixed(2)}
          {onlyExtras ? ", with extras" : ""}
        </h2>
        {picked.length === 0 ? (
          <p className="text-gray-600">No frames to show.</p>
        ) : (
          <>
            <p className="text-sm text-gray-600">
              showing {gallery.length} of {picked.length}
            </p>
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              {gallery.map((row, index) => (
                <DinoCard
                  key={row.run.run ?? `frame-${index}`}
                  row={row}
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

export const Sam3Page: FC = () => {
  const [threshold, setThreshold] = useState(THRESHOLD_DEFAULT);
  const [thresholdText, setThresholdText] = useState(
    THRESHOLD_DEFAULT.toFixed(2),
  );

  const { data, isPending, error } = useQuery({
    queryKey: ["dino"],
    queryFn: getDino,
    refetchInterval: REFETCH_MS,
    retry: false,
  });
  const dino = data ?? null;
  const dinoCount = dino?.runs?.length ?? 0;

  const runsDone = dino?.runs_done ?? dinoCount;
  const runsTotal = dino?.runs_total ?? dinoCount;
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
          Grounding DINO evaluation (temporary)
        </h1>
        <Button variant="outline" asChild>
          <Link to="/">← Back</Link>
        </Button>
      </div>

      {isPending && <div className="text-gray-600">Loading...</div>}

      {!isPending && error && (
        <div className="text-red-600">
          Failed to load dino.json: {error.message}
        </div>
      )}

      {!isPending && !error && !dino && (
        <div className="rounded-xl border border-slate-300 bg-slate-100/50 p-6 text-gray-600">
          dino data not published yet. This page retries every minute.
        </div>
      )}

      {!isPending && !error && dino && (
        <div className="flex flex-col gap-6">
          <div className="flex flex-col gap-2">
            <div className="flex flex-wrap items-baseline justify-between gap-2 text-sm text-gray-600">
              <span>updated {formatUpdated(dino.updated)}</span>
              <span className="font-mono">
                {runsDone} / {runsTotal} runs
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

          <DinoView dino={dino} threshold={threshold} />
        </div>
      )}
    </div>
  );
};
