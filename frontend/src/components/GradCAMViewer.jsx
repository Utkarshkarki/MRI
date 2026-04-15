import React, { useState } from 'react';
import './GradCAMViewer.css';

export default function GradCAMViewer({ heatmap_b64, originalPreview }) {
  const [mode, setMode] = useState('split'); // 'original' | 'heatmap' | 'split'

  return (
    <div className="gradcam glass-card animate-fadeUp-d1">
      <div className="gradcam__header">
        <div>
          <p className="section-label">Explainability — Grad-CAM</p>
          <h3 className="gradcam__title">Spatial Activation Heatmap</h3>
        </div>
        {/* Toggle */}
        <div className="gradcam__toggle" role="group" aria-label="View mode">
          {['original', 'split', 'heatmap'].map(m => (
            <button
              key={m}
              className={`gradcam__toggle-btn ${mode === m ? 'active' : ''}`}
              onClick={() => setMode(m)}
            >
              {m === 'original' ? 'MRI' : m === 'heatmap' ? 'Grad-CAM' : 'Side-by-Side'}
            </button>
          ))}
        </div>
      </div>

      <div className={`gradcam__viewer gradcam__viewer--${mode}`}>
        {(mode === 'original' || mode === 'split') && (
          <div className="gradcam__pane">
            <span className="gradcam__pane-label">Original MRI</span>
            {originalPreview
              ? <img src={originalPreview} alt="Original MRI scan" className="gradcam__img" />
              : <div className="gradcam__placeholder">Preview not available</div>
            }
          </div>
        )}
        {(mode === 'heatmap' || mode === 'split') && (
          <div className="gradcam__pane">
            <span className="gradcam__pane-label">Grad-CAM Overlay</span>
            <img
              src={`data:image/png;base64,${heatmap_b64}`}
              alt="Grad-CAM activation heatmap"
              className="gradcam__img"
            />
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="gradcam__legend">
        <span className="gradcam__legend-title">Activation Intensity →</span>
        <div className="gradcam__legend-bar" />
        <div className="gradcam__legend-labels">
          <span>Low</span><span>Medium</span><span>High</span>
        </div>
      </div>

      <p className="gradcam__note">
        Warmer regions (red/yellow) indicate areas the model weighted most heavily for its classification decision.
      </p>
    </div>
  );
}
