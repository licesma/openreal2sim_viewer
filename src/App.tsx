import "./App.css";
import "@google/model-viewer";
import { MainPage } from "@/pages/MainPage";
import { StatsPage } from "@/pages/StatsPage";
import { ManualMetadataPage } from "@/pages/ManualMetadataPage";
import { Sam3Page } from "@/pages/Sam3Page";
import { useMetadata } from "./hooks/useMetadata";
import type { AuthorType } from "@/types/Author";
import { type StatusType } from "@/types/Status";
import { type PoseType } from "@/types/Pose";
import { ALL_FEATURES, featureOptions, type FeatureType } from "@/types/Feature";
import { Separator } from "@/components/ui/separator";
import { ControlBar } from "@/components/ControlBar";
import type { WeekType } from "@/types/Week";
import { useState, useMemo } from "react";
import { Routes, Route } from "react-router-dom";
import { filterVideos } from "@/functions/filterVideos";

function App() {
  const [selectedAuthor, setSelectedAuthor] = useState<AuthorType>("all");
  const [selectedWeek, setSelectedWeek] = useState<WeekType>("all");
  const [selectedStatus, setSelectedStatus] = useState<StatusType>("all");
  const [selectedPose, setSelectedPose] = useState<PoseType>("all");
  const [selectedFeature, setSelectedFeature] = useState<FeatureType>(ALL_FEATURES);
  const [nameFilter, setNameFilter] = useState("");
  const [showFilters, setShowFilters] = useState(false);

  const { data } = useMetadata();

  const features = useMemo(() => (data ? featureOptions(data) : []), [data]);

  const filteredVideos = useMemo(() => {
    if (!data) return {};
    return filterVideos(
      data,
      selectedAuthor,
      selectedWeek,
      selectedStatus,
      selectedPose,
      selectedFeature,
      nameFilter
    );
  }, [data, selectedAuthor, selectedWeek, selectedStatus, selectedPose, selectedFeature, nameFilter]);

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="shrink-0 mb-2">
        <ControlBar showFilters={showFilters} onToggleFilters={() => setShowFilters(!showFilters)} />
        <Separator orientation="horizontal" />
      </div>
      <Routes>
        <Route
          path="/:videoId?"
          element={
            data ? (
              <div className="flex flex-col gap-4 h-full min-h-0 flex-1">
                <MainPage
                  allMetadata={filteredVideos}
                  selectedAuthor={selectedAuthor}
                  onAuthorChange={(author: AuthorType) => setSelectedAuthor(author)}
                  selectedWeek={selectedWeek}
                  onWeekChange={(week: WeekType) => setSelectedWeek(week)}
                  selectedStatus={selectedStatus}
                  onStatusChange={(status: StatusType) => setSelectedStatus(status)}
                  selectedPose={selectedPose}
                  onPoseChange={(p: PoseType) => setSelectedPose(p)}
                  features={features}
                  selectedFeature={selectedFeature}
                  onFeatureChange={(feature: FeatureType) => setSelectedFeature(feature)}
                  nameFilter={nameFilter}
                  onNameFilterChange={(name: string) => setNameFilter(name)}
                  showFilters={showFilters}
                />
              </div>
            ) : null
          }
        />
        <Route path="/stats" element={<StatsPage />} />
        <Route path="/manual_metadata" element={<ManualMetadataPage />} />
        <Route path="/sam3" element={<Sam3Page />} />
      </Routes>
    </div>
  );
}

export default App;
