export const ALL_FEATURES = "all";

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
  return [...found].sort((a, b) => a.localeCompare(b));
}
