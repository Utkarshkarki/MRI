import React, { useState, useRef, useCallback } from 'react';
import './ImageUploader.css';

export default function ImageUploader({ onResult, isLoading, setIsLoading, onPreviewChange }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState('');
  const inputRef = useRef(null);

  const ALLOWED = ['image/png', 'image/jpeg', 'image/jpg'];

  const handleFile = useCallback((f) => {
    if (!f) return;
    if (!ALLOWED.includes(f.type)) {
      setError('Please upload a PNG or JPG image.');
      return;
    }
    setError('');
    setFile(f);
    const objectUrl = URL.createObjectURL(f);
    setPreview(objectUrl);
    if (onPreviewChange) onPreviewChange(objectUrl);
  }, [onPreviewChange]);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const onDragOver = (e) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = () => setDragging(false);

  const handleSubmit = async () => {
    if (!file) return;
    setIsLoading(true);
    onResult(null);
    setError('');

    try {
      const form = new FormData();
      form.append('file', file);

      const res = await fetch('http://127.0.0.1:8000/api/predict', {
        method: 'POST',
        body: form,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Server error');
      }

      const data = await res.json();
      onResult(data);
    } catch (e) {
      setError(`Analysis failed: ${e.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setError('');
    onResult(null);
  };

  return (
    <section id="uploader" className="uploader-section glass-card animate-fadeUp-d1">
      <p className="section-label">Step 1 — Upload MRI Scan</p>
      <h2 className="uploader-title">Patient Image Intake</h2>

      {/* Drop Zone */}
      <div
        className={`drop-zone ${dragging ? 'drop-zone--active' : ''} ${preview ? 'drop-zone--has-file' : ''}`}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => !preview && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        aria-label="MRI image upload zone"
      >
        <input
          ref={inputRef}
          type="file"
          accept=".png,.jpg,.jpeg"
          onChange={(e) => handleFile(e.target.files[0])}
          style={{ display: 'none' }}
          id="mri-file-input"
        />

        {preview ? (
          <div className="drop-zone__preview">
            <img src={preview} alt="MRI preview" className="drop-zone__preview-img" />
            <div className="drop-zone__preview-overlay">
              <span className="drop-zone__filename">{file?.name}</span>
              <span className="drop-zone__filesize">
                {(file?.size / 1024).toFixed(1)} KB
              </span>
            </div>
            {/* Scan line animation */}
            <div className="drop-zone__scanline" />
          </div>
        ) : (
          <div className="drop-zone__empty">
            <div className="drop-zone__icon">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <p className="drop-zone__title">Drop MRI scan here</p>
            <p className="drop-zone__sub">or click to browse · PNG, JPG supported</p>
            <div className="drop-zone__formats">
              <span>PNG</span><span>JPG</span><span>JPEG</span>
            </div>
          </div>
        )}
      </div>

      {/* Error message */}
      {error && (
        <div className="uploader-error" role="alert">
          <span>⚠</span> {error}
        </div>
      )}

      {/* Action buttons */}
      <div className="uploader-actions">
        {preview && (
          <button className="btn-ghost" onClick={reset} disabled={isLoading}>
            ✕ Clear
          </button>
        )}
        <button
          id="analyze-btn"
          className="btn-primary"
          onClick={handleSubmit}
          disabled={!file || isLoading}
        >
          {isLoading ? (
            <>
              <span className="spinner" />
              Analyzing…
            </>
          ) : (
            <>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
              </svg>
              Run Diagnostics
            </>
          )}
        </button>
      </div>
    </section>
  );
}
