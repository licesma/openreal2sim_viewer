// @ts-nocheck

export function ModelViewer({
  src,
  alt,
  orientation,
  cameraOrbit = "150deg 90deg 2m",
}: {
  src: string;
  alt: string;
  orientation?: string;
  cameraOrbit?: string;
}) {
  return (
    <model-viewer
      src={src}
      alt={alt}
      orientation={orientation}
      camera-controls
      rotation="0 0 0"
      rotation-per-second="9deg"
      interaction-prompt="none"
      camera-target="50% 50% 50%"
      camera-orbit={cameraOrbit}
      style={{
        width: "100%",
        height: "100%",
        backgroundColor: "#dae7f7ff",
      }}
    ></model-viewer>
  );
}
