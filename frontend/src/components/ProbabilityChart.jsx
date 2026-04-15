import React from 'react';
import './ProbabilityChart.css';

const CLASS_COLORS = {
  glioma:     '#ff4757',
  meningioma: '#ffa502',
  pituitary:  '#a29bfe',
  notumor:    '#2ed573',
};

const CLASS_LABELS = {
  glioma:     'Glioma',
  meningioma: 'Meningioma',
  pituitary:  'Pituitary Tumor',
  notumor:    'No Tumor',
};

export default function ProbabilityChart({ probabilities, diagnosis }) {
  const entries = Object.entries(probabilities).sort(([,a],[,b]) => b - a);

  return (
    <div className="probchart glass-card animate-fadeUp-d2">
      <p className="section-label">Class Probabilities</p>
      <h3 className="probchart__title">Softmax Distribution</h3>

      <div className="probchart__bars">
        {entries.map(([cls, prob]) => {
          const pct = (prob * 100).toFixed(2);
          const isTop = cls === diagnosis;
          const color = CLASS_COLORS[cls] || '#00d4ff';

          return (
            <div key={cls} className={`probchart__row ${isTop ? 'probchart__row--top' : ''}`}>
              <div className="probchart__row-header">
                <span className="probchart__cls-label">
                  {isTop && <span className="probchart__top-dot" style={{ background: color }} />}
                  {CLASS_LABELS[cls]}
                </span>
                <span className="probchart__pct mono" style={{ color: isTop ? color : undefined }}>
                  {pct}%
                </span>
              </div>
              <div className="probchart__track">
                <div
                  className="probchart__fill"
                  style={{
                    width: `${pct}%`,
                    background: isTop
                      ? `linear-gradient(90deg, ${color}99, ${color})`
                      : 'rgba(255,255,255,0.12)',
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <p className="probchart__note">
        Values represent averaged softmax probabilities across 15 Monte Carlo forward passes.
      </p>
    </div>
  );
}
