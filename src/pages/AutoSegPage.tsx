import { useCallback, useEffect, useRef, useState } from "react";
import type { FC } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";

const BUCKET_URL = "https://openreal2sim-sam.s3.us-west-2.amazonaws.com/";
const INDEX_URL = `${BUCKET_URL}auto_seg/index.json`;
const PRELOAD_AHEAD = 5;
const PLAY_INTERVAL_MS = 100;

interface AutoSegObject {
  name?: string;
  iou?: number;
}

interface AutoSegRun {
  id: string;
  frame0: string;
  week?: string;
  author?: string;
  run?: string;
  gt_objects?: string[];
  iou_frame0?: number;
  per_object?: AutoSegObject[];
  n_frames?: number;
  frames_prefix?: string;
  methods?: string[];
  iou_video?: Record<string, number>;
}

interface AutoSegIndex {
  runs: AutoSegRun[];
}

async function getAutoSegIndex(): Promise<AutoSegIndex> {
  const response = await fetch(INDEX_URL, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to fetch auto_seg index (${response.status})`);
  }
  return (await response.json()) as AutoSegIndex;
}

function formatIou(value?: number): string {
  return typeof value === "number" ? value.toFixed(2) : "n/a";
}

function frameCountOf(run: AutoSegRun): number {
  if (!run.frames_prefix || typeof run.n_frames !== "number") {
    return 1;
  }
  return Math.max(1, Math.floor(run.n_frames));
}

function frameUrlOf(run: AutoSegRun, index: number): string {
  if (!run.frames_prefix || typeof run.n_frames !== "number") {
    return `${BUCKET_URL}${run.frame0}`;
  }
  const padded = String(index).padStart(6, "0");
  return `${BUCKET_URL}${run.frames_prefix}${padded}.png`;
}

const FrameViewer: FC<{ run: AutoSegRun }> = ({ run }) => {
  const frameCount = frameCountOf(run);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const activeRef = useRef(false);

  const step = useCallback(
    (delta: number) => {
      setFrame((current) => {
        const next = current + delta;
        if (next < 0) return 0;
        if (next > frameCount - 1) return frameCount - 1;
        return next;
      });
    },
    [frameCount],
  );

  // Keep the current frame inside the range when the index changes.
  useEffect(() => {
    setFrame((current) => Math.min(current, frameCount - 1));
  }, [frameCount]);

  // Play loop at roughly 10 fps, looping back to the first frame.
  useEffect(() => {
    if (!playing || frameCount < 2) return;
    const timer = window.setInterval(() => {
      setFrame((current) => (current + 1) % frameCount);
    }, PLAY_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [playing, frameCount]);

  // Preload the next few frames so scrubbing forward stays smooth.
  useEffect(() => {
    if (frameCount < 2) return;
    for (let offset = 1; offset <= PRELOAD_AHEAD; offset += 1) {
      const next = frame + offset;
      if (next > frameCount - 1) break;
      const image = new Image();
      image.src = frameUrlOf(run, next);
    }
  }, [run, frame, frameCount]);

  // Arrow keys move one frame while the card is hovered or focused.
  useEffect(() => {
    if (frameCount < 2) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (!activeRef.current) return;
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      const target = event.target as HTMLElement | null;
      if (target && target.tagName === "INPUT") return;
      event.preventDefault();
      step(event.key === "ArrowRight" ? 1 : -1);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [frameCount, step]);

  const setActive = (value: boolean) => {
    activeRef.current = value;
  };

  return (
    <div
      className="flex flex-col gap-2 outline-none"
      tabIndex={0}
      onMouseEnter={() => setActive(true)}
      onMouseLeave={() => setActive(false)}
      onFocus={() => setActive(true)}
      onBlur={() => setActive(false)}
    >
      {/* Three panels baked into the PNG: GT, Grounding DINO + SAM2, SAM 3 */}
      <img
        className="w-full rounded-lg border border-slate-300 bg-black"
        src={frameUrlOf(run, frame)}
        alt={`Segmentation comparison for ${run.id}, frame ${frame}`}
      />

      {frameCount > 1 ? (
        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setPlaying((current) => !current)}
          >
            {playing ? "Pause" : "Play"}
          </Button>
          <input
            className="h-2 flex-1 cursor-pointer accent-slate-700"
            type="range"
            min={0}
            max={frameCount - 1}
            step={1}
            value={frame}
            onChange={(event) => setFrame(Number(event.target.value))}
            aria-label={`Frame for ${run.id}`}
          />
          <span className="w-24 shrink-0 text-right font-mono text-sm text-gray-600">
            {frame} / {frameCount - 1}
          </span>
        </div>
      ) : (
        <p className="text-sm text-gray-500">
          Frame 0 only, no frame sequence in the index yet.
        </p>
      )}
    </div>
  );
};

const RunCard: FC<{ run: AutoSegRun }> = ({ run }) => {
  const perObject = run.per_object ?? [];
  const videoIous = Object.entries(run.iou_video ?? {});

  return (
    <div className="flex w-full flex-col gap-3 rounded-xl border border-slate-300 bg-slate-100/50 p-4 shadow-lg backdrop-blur-sm">
      <div className="flex flex-col gap-1">
        <h2 className="text-lg font-semibold break-all text-gray-800">
          {run.id}
        </h2>
        <p className="text-sm text-gray-600">
          GT objects:{" "}
          {run.gt_objects && run.gt_objects.length > 0
            ? run.gt_objects.join(", ")
            : "n/a"}
        </p>
        <p className="text-sm font-medium text-gray-700">
          IoU frame 0: {formatIou(run.iou_frame0)}
        </p>
        {videoIous.length > 0 && (
          <p className="text-sm font-medium text-gray-700">
            IoU video:{" "}
            {videoIous
              .map(([method, value]) => `${method} ${formatIou(value)}`)
              .join(", ")}
          </p>
        )}
      </div>

      <FrameViewer run={run} />

      {perObject.length > 0 && (
        <ul className="flex flex-col gap-0.5 text-sm text-gray-600">
          {perObject.map((object, index) => (
            <li key={`${object.name ?? "unnamed"}-${index}`}>
              {object.name ?? "unnamed"}: {formatIou(object.iou)}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export const AutoSegPage: FC = () => {
  const { data, isPending, error } = useQuery({
    queryKey: ["autoSegIndex"],
    queryFn: getAutoSegIndex,
    retry: false,
  });

  const runs = data?.runs ?? [];

  return (
    <div className="min-h-screen p-8">
      <div className="mb-8 flex items-center justify-between">
        <h1 className="text-3xl font-bold tracking-tight text-gray-800">
          Automatic segmentation (temporary)
        </h1>
        <Button variant="outline" asChild>
          <Link to="/">← Back</Link>
        </Button>
      </div>

      {isPending && <div>Loading...</div>}

      {!isPending && error && (
        <div className="text-red-600">
          Failed to load auto_seg index: {error.message}
        </div>
      )}

      {!isPending && !error && runs.length === 0 && (
        <div className="text-gray-600">No runs in the index.</div>
      )}

      {!isPending && !error && runs.length > 0 && (
        <div className="flex flex-col gap-6">
          {runs.map((run) => (
            <RunCard key={run.id} run={run} />
          ))}
        </div>
      )}
    </div>
  );
};
