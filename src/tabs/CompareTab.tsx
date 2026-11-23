import "../App.css";
import "@google/model-viewer";
import type { FC, ReactNode } from "react";
import { ModelViewer } from "@/components/ModelViewer";
import { COMPARE_BASE_URL, COMPARE_RUNS } from "@/constants";

// Both scenes are tabletops: look down on them and let model-viewer frame them.
const COMPARE_ORBIT = "0deg 60deg auto";

interface CompareTabProps {
  selectedVideo: string | undefined;
}

const Panel: FC<{ title: string; children: ReactNode }> = ({
  title,
  children,
}) => (
  <div className="flex flex-col min-h-0 border border-gray-300 rounded-lg overflow-hidden">
    <div className="h-8 flex items-center px-3 text-sm font-semibold shrink-0">
      {title}
    </div>
    <div className="flex-1 min-h-0">{children}</div>
  </div>
);

const CompareTab: FC<CompareTabProps> = ({ selectedVideo }) => {
  const compareRun = COMPARE_RUNS.find((entry) => entry.run === selectedVideo);

  if (!compareRun) {
    return (
      <div className="w-full h-full flex items-center justify-center text-center text-2xl">
        Select one of the agent vs pipeline runs
      </div>
    );
  }

  const { run, pipelinePrefix, agentOrientation } = compareRun;
  const pipelineBase = `${COMPARE_BASE_URL}/${pipelinePrefix}/${run}`;
  const agentBase = `${COMPARE_BASE_URL}/agent/${run}`;

  return (
    <div className="h-full grid grid-cols-2 grid-rows-2 gap-2">
      <Panel title="Photo">
        <img
          src={`${pipelineBase}/source/000000.jpg`}
          alt="Source frame"
          className="w-full h-full object-contain"
        />
      </Panel>
      <Panel title="Agent render">
        <img
          src={`${agentBase}/agent/render.png`}
          alt="Agent render of the scene"
          className="w-full h-full object-contain"
        />
      </Panel>
      <Panel title="Pipeline">
        <ModelViewer
          src={`${pipelineBase}/reconstruction/scene.glb`}
          alt="Pipeline reconstruction"
          cameraOrbit={COMPARE_ORBIT}
        />
      </Panel>
      <Panel title="Agent">
        <ModelViewer
          src={`${agentBase}/reconstruction/scene.glb`}
          alt="Agent reconstruction"
          orientation={agentOrientation}
          cameraOrbit={COMPARE_ORBIT}
        />
      </Panel>
    </div>
  );
};

export default CompareTab;
