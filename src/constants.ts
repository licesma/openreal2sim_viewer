export const FIREBASE_COLLECTION = "reconstructions";
export const SAM_DOCUMENT = "sam";

// Colors
export const LIGHT_BLUE = "#dae7f7";
export const DARK_BLUE = "#1e3a8a";

// CloudFront distribution in front of the openreal2sim-sam bucket.
export const COMPARE_BASE_URL = "https://d3bztodlk4kt2m.cloudfront.net";

// Agent scenes are metric and Z up. model-viewer renders Y up, so a scene that
// does not carry its own conversion node needs a -90 deg pitch, given as
// "roll pitch yaw".
export const ZUP_ORIENTATION = "0deg -90deg 0deg";

export interface CompareRun {
  run: string;
  pipelinePrefix: string;
  agentOrientation?: string;
}

// Agent vs pipeline experiment. Every run has a pipeline reconstruction under
// <week>/<author>/<run>/ and an agent one under agent/<run>/.
export const COMPARE_RUNS: CompareRun[] = [
  {
    run: "scene_00119",
    pipelinePrefix: "week_2/divleen",
    agentOrientation: ZUP_ORIENTATION,
  },
  // Its scene.glb already has a scene_zup root node, so no orientation here.
  { run: "scene_00597", pipelinePrefix: "week_6/divleen" },
  // Agent scene not uploaded yet; its up axis still has to be checked.
  { run: "video_00941", pipelinePrefix: "week_10/junsoo" },
];
