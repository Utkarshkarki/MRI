import React, { useEffect, useRef } from 'react';
import DiagnosisCard from './DiagnosisCard';
import GradCAMViewer from './GradCAMViewer';
import ProbabilityChart from './ProbabilityChart';
import ClinicalSummary from './ClinicalSummary';
import './ResultsPanel.css';

export default function ResultsPanel({ results, originalPreview }) {
  const panelRef = useRef(null);

  useEffect(() => {
    // Scroll into view on load
    if (results && panelRef.current) {
      setTimeout(() => {
        panelRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    }
  }, [results]);

  if (!results) return null;

  return (
    <div className="results-panel" ref={panelRef}>
      <div className="results-panel__grid">
        {/* Left Column */}
        <div className="results-panel__col results-panel__col--left">
          <DiagnosisCard
            diagnosis={results.diagnosis}
            confidence={results.confidence}
            uncertainty={results.uncertainty}
            is_anomaly={results.is_anomaly}
          />
          <GradCAMViewer
            heatmap_b64={results.heatmap_b64}
            originalPreview={originalPreview}
          />
        </div>

        {/* Right Column */}
        <div className="results-panel__col results-panel__col--right">
          <ProbabilityChart
            probabilities={results.probabilities}
            diagnosis={results.diagnosis}
          />
          <ClinicalSummary
            diagnosis={results.diagnosis}
            confidence={results.confidence}
            uncertainty={results.uncertainty}
            is_anomaly={results.is_anomaly}
          />
        </div>
      </div>
    </div>
  );
}
