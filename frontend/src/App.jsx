import React, { useState } from 'react';
import HeroSection from './components/HeroSection';
import ImageUploader from './components/ImageUploader';
import ResultsPanel from './components/ResultsPanel';

function App() {
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [originalPreview, setOriginalPreview] = useState(null);

  const handleResult = (data) => {
    setResults(data);
  };

  const handlePreviewChange = (previewUrl) => {
    setOriginalPreview(previewUrl);
  };

  return (
    <div className="app-wrapper">
      <HeroSection />
      
      <main>
        <ImageUploader 
          onResult={handleResult} 
          isLoading={isLoading} 
          setIsLoading={setIsLoading} 
          onPreviewChange={handlePreviewChange}
        />
        
        <ResultsPanel 
          results={results} 
          originalPreview={originalPreview}
        />
      </main>
    </div>
  );
}

export default App;
