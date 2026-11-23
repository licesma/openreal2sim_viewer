export const ALL_FEATURES = "all";

// Not a metadata value: it selects the runs of the agent vs pipeline
// experiment and shows both reconstructions side by side.
export const AGENT_COMPARE = "Agent vs pipeline";

export type FeatureType = string;

export function featureLabel(feature: FeatureType): string {
  return feature === ALL_FEATURES ? "All" : feature;
}

// Every distinct non-empty feature value present in the metadata, sorted.
export function featureOptions(
  allMetadata: Record<string, { feature?: string }>
): FeatureType[] {
  const found = new Set<string>();
  for (const meta of Object.values(allMetadata)) {
    const feature = meta?.feature;
    if (typeof feature === "string" && feature.trim() !== "") {
      found.add(feature);
    }
  }
  found.add(AGENT_COMPARE);
  return [...found].sort((a, b) => a.localeCompare(b));
}
