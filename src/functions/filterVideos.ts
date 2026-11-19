import "../App.css";
import "@google/model-viewer";
import type { VideoMetadata } from "@/types/VideoMetadata";
import type { AuthorType } from "@/types/Author";
import type { WeekType } from "@/types/Week";
import { ALL_FEATURES } from "@/types/Feature";

export function filterVideos(
    allMetadata: Record<string, VideoMetadata>,
    selectedAuthor: AuthorType,
    selectedWeek: WeekType,
    selectedStatus: string,
    selectedPose: string,
    selectedFeature: string = ALL_FEATURES,
    nameFilter: string = ""
) {
    const filteredByAuthor =
        selectedAuthor !== "all"
            ? (Object.fromEntries(
                Object.entries(allMetadata).filter(
                    ([_name, meta]) => meta.author == selectedAuthor
                )
            ) as Record<string, VideoMetadata>)
            : allMetadata;

    const filteredByWeek =
        selectedWeek !== "all"
            ? (Object.fromEntries(
                Object.entries(filteredByAuthor).filter(
                    ([_name, meta]) => meta.week == selectedWeek
                )
            ) as Record<string, VideoMetadata>)
            : filteredByAuthor;

    const filteredByStatus =
        selectedStatus !== "all"
            ? (Object.fromEntries(
                Object.entries(filteredByWeek).filter(
                    ([_name, meta]) => meta.status == selectedStatus
                )
            ) as Record<string, VideoMetadata>)
            : filteredByWeek;

    const filteredByPose =
        selectedPose !== "all"
            ? (Object.fromEntries(
                Object.entries(filteredByStatus).filter(
                    ([_name, meta]) => meta.pose == selectedPose
                )
            ) as Record<string, VideoMetadata>)
            : filteredByStatus;

    const filteredByFeature =
        selectedFeature !== ALL_FEATURES
            ? (Object.fromEntries(
                Object.entries(filteredByPose).filter(
                    ([_name, meta]) => meta.feature == selectedFeature
                )
            ) as Record<string, VideoMetadata>)
            : filteredByPose;

    const nameTerms = nameFilter.trim().toLowerCase().split(/\s+/).filter(Boolean);

    const filteredByName =
        nameTerms.length > 0
            ? (Object.fromEntries(
                Object.entries(filteredByFeature).filter(([name]) =>
                    nameTerms.some((term) => name.toLowerCase().includes(term))
                )
            ) as Record<string, VideoMetadata>)
            : filteredByFeature;

    return filteredByName;
}