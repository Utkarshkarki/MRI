import React from 'react';
import './HeroSection.css';

export default function HeroSection() {
  return (
    <header className="hero">
      {/* Ambient glow blobs */}
      <div className="hero__blob hero__blob--1" />
      <div className="hero__blob hero__blob--2" />

      <div className="hero__content animate-fadeUp">
        {/* Brain icon with pulse ring */}
        <div className="hero__icon-wrap">
          <div className="hero__pulse-ring" />
          <div className="hero__pulse-ring hero__pulse-ring--delay" />
          <span className="hero__icon">🧠</span>
        </div>

        <p className="section-label">AI-Powered Neuro-Diagnostics</p>

        <h1 className="hero__title">
          Clinical-Grade
          <span className="hero__title-gradient"> Brain Tumor</span>
          <br />Analysis System
        </h1>

        <p className="hero__subtitle">
          Powered by <strong>MCDropoutResNet</strong> · Grad-CAM Explainability ·
          Monte Carlo Uncertainty Quantification
        </p>

        {/* Tech badges */}
        <div className="hero__badges">
          {['ResNet-50', 'MC Dropout', 'Grad-CAM', 'Focal Loss', 'CUDA'].map(b => (
            <span key={b} className="badge badge-neutral">{b}</span>
          ))}
        </div>

        {/* CTA */}
        <a href="#uploader" className="btn-primary hero__cta">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <polyline points="16 16 12 12 8 16" />
            <line x1="12" y1="12" x2="12" y2="21" />
            <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3" />
          </svg>
          Upload MRI Scan
        </a>
      </div>

      {/* Stats strip */}
      <div className="hero__stats animate-fadeUp-d2">
        {[
          { label: 'Tumor Classes', value: '4' },
          { label: 'MC Passes', value: '15' },
          { label: 'Backbone', value: 'ResNet-50' },
          { label: 'Explainability', value: 'Grad-CAM' },
        ].map(s => (
          <div key={s.label} className="hero__stat">
            <span className="hero__stat-value">{s.value}</span>
            <span className="hero__stat-label">{s.label}</span>
          </div>
        ))}
      </div>
    </header>
  );
}
