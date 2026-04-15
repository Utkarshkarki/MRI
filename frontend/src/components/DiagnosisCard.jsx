import React from 'react';
import './DiagnosisCard.css';

const CLASS_META = {
  glioma:     { label: 'Glioma',          icon: '🔴', type: 'danger' },
  meningioma: { label: 'Meningioma',      icon: '🟠', type: 'danger' },
  pituitary:  { label: 'Pituitary Tumor', icon: '🟣', type: 'danger' },
  notumor:    { label: 'No Tumor',        icon: '🟢', type: 'safe'   },
};

export default function DiagnosisCard({ diagnosis, confidence, uncertainty, is_anomaly }) {
  const meta = CLASS_META[diagnosis] || { label: diagnosis, icon: '⚪', type: 'neutral' };
  const isSafe = meta.type === 'safe';
  const confPct = (confidence * 100).toFixed(1);
  const uncPct  = Math.min(uncertainty / 1.5, 1); // normalize entropy to 0-1 range for display

  return (
    <div className={`diag-card glass-card animate-fadeUp ${isSafe ? 'diag-card--safe' : 'diag-card--danger'}`}>
      <p className="section-label">Diagnostic Result</p>

      {/* Main diagnosis badge */}
      <div className="diag-card__main">
        <span className="diag-card__icon">{meta.icon}</span>
        <div>
          <h2 className={`diag-card__label ${isSafe ? 'text-safe' : 'text-danger'}`}>
            {meta.label.toUpperCase()}
          </h2>
          <span className={`badge ${isSafe ? 'badge-safe' : 'badge-danger'}`}>
            {isSafe ? '✓ No Pathology Detected' : '⚠ Tumor Detected'}
          </span>
        </div>
      </div>

      <div className="divider" />

      {/* Confidence */}
      <div className="diag-card__metric">
        <div className="diag-card__metric-header">
          <span className="diag-card__metric-label">Model Confidence</span>
          <span className="diag-card__metric-value mono">{confPct}%</span>
        </div>
        <div className="diag-card__bar-track">
          <div
            className={`diag-card__bar-fill ${isSafe ? 'bar-safe' : 'bar-danger'}`}
            style={{ width: `${confPct}%` }}
          />
        </div>
      </div>

      {/* Uncertainty */}
      <div className="diag-card__metric">
        <div className="diag-card__metric-header">
          <span className="diag-card__metric-label">Uncertainty (Shannon Entropy)</span>
          <span className="diag-card__metric-value mono">{uncertainty.toFixed(4)}</span>
        </div>
        <div className="diag-card__bar-track">
          <div
            className={`diag-card__bar-fill ${is_anomaly ? 'bar-warn' : 'bar-accent'}`}
            style={{ width: `${(uncPct * 100).toFixed(1)}%` }}
          />
        </div>
      </div>

      {/* Anomaly alert */}
      {is_anomaly && (
        <div className="diag-card__anomaly" role="alert">
          <span>🚨</span>
          <div>
            <strong>Clinical Anomaly Triggered</strong>
            <p>High uncertainty detected. Manual review by a radiologist is strongly recommended.</p>
          </div>
        </div>
      )}
    </div>
  );
}
