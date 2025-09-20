import React, { useState, useRef, useCallback, useEffect } from 'react';

const ApexCurveEditor = () => {
  const svgRef = useRef(null);
  const [activeChannel, setActiveChannel] = useState('master');
  const [isDragging, setIsDragging] = useState(false);
  const [dragPointIndex, setDragPointIndex] = useState(-1);
  
  // Curve points for each channel (x, y coordinates in 0-255 range)
  const [curves, setCurves] = useState({
    master: [[0, 0], [64, 64], [128, 128], [192, 192], [255, 255]],
    red: [[0, 0], [64, 64], [128, 128], [192, 192], [255, 255]],
    green: [[0, 0], [64, 64], [128, 128], [192, 192], [255, 255]],
    blue: [[0, 0], [64, 64], [128, 128], [192, 192], [255, 255]]
  });

  const [previewSettings, setPreviewSettings] = useState({
    blendMode: 'normal',
    opacity: 1.0,
    preserveLuminance: false,
    showHistogram: true
  });

  const chartSize = 300;
  const padding = 40;

  // Convert curve coordinates to SVG coordinates
  const toSVG = useCallback((x, y) => {
    return {
      x: padding + (x / 255) * chartSize,
      y: padding + chartSize - (y / 255) * chartSize
    };
  }, []);

  // Convert SVG coordinates back to curve coordinates
  const fromSVG = useCallback((svgX, svgY) => {
    return {
      x: Math.max(0, Math.min(255, ((svgX - padding) / chartSize) * 255)),
      y: Math.max(0, Math.min(255, ((padding + chartSize - svgY) / chartSize) * 255))
    };
  }, []);

  // Generate smooth curve path using cubic bezier
  const generateCurvePath = useCallback((points) => {
    if (points.length < 2) return '';
    
    let path = '';
    const svgPoints = points.map(([x, y]) => toSVG(x, y));
    
    path += `M ${svgPoints[0].x} ${svgPoints[0].y}`;
    
    for (let i = 1; i < svgPoints.length; i++) {
      const prev = svgPoints[i - 1];
      const curr = svgPoints[i];
      const next = svgPoints[i + 1];
      
      if (i === 1) {
        // First curve segment
        const cp1x = prev.x + (curr.x - prev.x) * 0.3;
        const cp1y = prev.y;
        const cp2x = curr.x - (curr.x - prev.x) * 0.3;
        const cp2y = curr.y;
        path += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${curr.x} ${curr.y}`;
      } else if (i === svgPoints.length - 1) {
        // Last curve segment
        const cp1x = prev.x + (curr.x - prev.x) * 0.3;
        const cp1y = prev.y;
        const cp2x = curr.x - (curr.x - prev.x) * 0.3;
        const cp2y = curr.y;
        path += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${curr.x} ${curr.y}`;
      } else {
        // Middle segments with smooth transitions
        const prevDx = curr.x - svgPoints[i - 2].x;
        const prevDy = curr.y - svgPoints[i - 2].y;
        const nextDx = next.x - prev.x;
        const nextDy = next.y - prev.y;
        
        const cp1x = prev.x + prevDx * 0.2;
        const cp1y = prev.y + prevDy * 0.2;
        const cp2x = curr.x - nextDx * 0.2;
        const cp2y = curr.y - nextDy * 0.2;
        
        path += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${curr.x} ${curr.y}`;
      }
    }
    
    return path;
  }, [toSVG]);

  // Handle mouse events for dragging points
  const handleMouseDown = useCallback((e, pointIndex) => {
    e.preventDefault();
    setIsDragging(true);
    setDragPointIndex(pointIndex);
  }, []);

  const handleMouseMove = useCallback((e) => {
    if (!isDragging || dragPointIndex === -1) return;
    
    const rect = svgRef.current.getBoundingClientRect();
    const svgX = e.clientX - rect.left;
    const svgY = e.clientY - rect.top;
    const newPos = fromSVG(svgX, svgY);
    
    setCurves(prev => {
      const newCurves = { ...prev };
      const currentCurve = [...newCurves[activeChannel]];
      
      // Constrain X position based on neighboring points
      if (dragPointIndex > 0 && dragPointIndex < currentCurve.length - 1) {
        const minX = currentCurve[dragPointIndex - 1][0] + 1;
        const maxX = currentCurve[dragPointIndex + 1][0] - 1;
        newPos.x = Math.max(minX, Math.min(maxX, newPos.x));
      } else if (dragPointIndex === 0) {
        newPos.x = 0; // Lock first point X
      } else if (dragPointIndex === currentCurve.length - 1) {
        newPos.x = 255; // Lock last point X
      }
      
      currentCurve[dragPointIndex] = [Math.round(newPos.x), Math.round(newPos.y)];
      newCurves[activeChannel] = currentCurve;
      
      return newCurves;
    });
  }, [isDragging, dragPointIndex, activeChannel, fromSVG]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    setDragPointIndex(-1);
  }, []);

  // Add point on curve click
  const handleCurveClick = useCallback((e) => {
    if (isDragging) return;
    
    const rect = svgRef.current.getBoundingClientRect();
    const svgX = e.clientX - rect.left;
    const svgY = e.clientY - rect.top;
    const newPos = fromSVG(svgX, svgY);
    
    setCurves(prev => {
      const newCurves = { ...prev };
      const currentCurve = [...newCurves[activeChannel]];
      
      // Find insertion point
      let insertIndex = currentCurve.length;
      for (let i = 0; i < currentCurve.length; i++) {
        if (newPos.x < currentCurve[i][0]) {
          insertIndex = i;
          break;
        }
      }
      
      currentCurve.splice(insertIndex, 0, [Math.round(newPos.x), Math.round(newPos.y)]);
      newCurves[activeChannel] = currentCurve;
      
      return newCurves;
    });
  }, [isDragging, activeChannel, fromSVG]);

  // Remove point (double-click, but not first/last)
  const handlePointDoubleClick = useCallback((e, pointIndex) => {
    e.stopPropagation();
    if (pointIndex === 0 || pointIndex === curves[activeChannel].length - 1) return;
    
    setCurves(prev => {
      const newCurves = { ...prev };
      const currentCurve = [...newCurves[activeChannel]];
      currentCurve.splice(pointIndex, 1);
      newCurves[activeChannel] = currentCurve;
      return newCurves;
    });
  }, [activeChannel, curves]);

  // Preset curves
  const applyPreset = useCallback((preset) => {
    const presets = {
      linear: [[0, 0], [64, 64], [128, 128], [192, 192], [255, 255]],
      slight_s: [[0, 0], [48, 32], [128, 128], [208, 224], [255, 255]],
      strong_s: [[0, 0], [32, 16], [128, 128], [224, 240], [255, 255]],
      brighten: [[0, 0], [80, 96], [160, 192], [224, 240], [255, 255]],
      darken: [[0, 0], [32, 16], [96, 64], [176, 144], [255, 255]],
      contrast: [[0, 0], [40, 16], [128, 128], [216, 240], [255, 255]],
      low_contrast: [[0, 0], [76, 96], [128, 128], [180, 160], [255, 255]],
      crushed_blacks: [[0, 32], [80, 80], [128, 128], [192, 192], [255, 255]],
      lifted_blacks: [[0, 0], [96, 128], [144, 160], [200, 216], [255, 255]]
    };
    
    if (presets[preset]) {
      setCurves(prev => ({
        ...prev,
        [activeChannel]: [...presets[preset]]
      }));
    }
  }, [activeChannel]);

  // Reset current channel
  const resetChannel = useCallback(() => {
    setCurves(prev => ({
      ...prev,
      [activeChannel]: [[0, 0], [64, 64], [128, 128], [192, 192], [255, 255]]
    }));
  }, [activeChannel]);

  // Add mouse event listeners
  useEffect(() => {
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [handleMouseMove, handleMouseUp]);

  const currentCurve = curves[activeChannel];
  const channelColors = {
    master: '#ffffff',
    red: '#ff4444',
    green: '#44ff44',
    blue: '#4444ff'
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-gray-900 text-white rounded-lg">
      <h2 className="text-2xl font-bold mb-6 text-center">Apex RGB Curve Editor</h2>
      
      {/* Channel Selector */}
      <div className="mb-6">
        <div className="flex gap-2 mb-4">
          {Object.keys(curves).map(channel => (
            <button
              key={channel}
              onClick={() => setActiveChannel(channel)}
              className={`px-4 py-2 rounded capitalize font-medium transition-all ${
                activeChannel === channel 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
              style={activeChannel === channel ? { backgroundColor: channelColors[channel] + '40', borderColor: channelColors[channel] } : {}}
            >
              {channel}
            </button>
          ))}
        </div>
        
        {/* Preset Buttons */}
        <div className="flex flex-wrap gap-2 mb-4">
          <button onClick={() => applyPreset('linear')} className="px-3 py-1 bg-gray-700 rounded text-sm hover:bg-gray-600">Linear</button>
          <button onClick={() => applyPreset('slight_s')} className="px-3 py-1 bg-gray-700 rounded text-sm hover:bg-gray-600">Slight S</button>
          <button onClick={() => applyPreset('strong_s')} className="px-3 py-1 bg-gray-700 rounded text-sm hover:bg-gray-600">Strong S</button>
          <button onClick={() => applyPreset('brighten')} className="px-3 py-1 bg-gray-700 rounded text-sm hover:bg-gray-600">Brighten</button>
          <button onClick={() => applyPreset('darken')} className="px-3 py-1 bg-gray-700 rounded text-sm hover:bg-gray-600">Darken</button>
          <button onClick={() => applyPreset('contrast')} className="px-3 py-1 bg-gray-700 rounded text-sm hover:bg-gray-600">Contrast</button>
          <button onClick={resetChannel} className="px-3 py-1 bg-red-600 rounded text-sm hover:bg-red-700">Reset</button>
        </div>
      </div>

      {/* Main Curve Editor */}
      <div className="flex gap-6">
        <div className="flex-1">
          <div className="bg-gray-800 p-4 rounded-lg">
            <svg
              ref={svgRef}
              width={chartSize + padding * 2}
              height={chartSize + padding * 2}
              className="border border-gray-600 bg-gray-900 cursor-crosshair"
              onClick={handleCurveClick}
            >
              {/* Grid */}
              <defs>
                <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                  <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#444" strokeWidth="0.5"/>
                </pattern>
              </defs>
              <rect x={padding} y={padding} width={chartSize} height={chartSize} fill="url(#grid)"/>
              
              {/* Diagonal reference line */}
              <line 
                x1={padding} 
                y1={padding + chartSize} 
                x2={padding + chartSize} 
                y2={padding}
                stroke="#666" 
                strokeWidth="1" 
                strokeDasharray="5,5"
              />
              
              {/* Curve */}
              <path
                d={generateCurvePath(currentCurve)}
                fill="none"
                stroke={channelColors[activeChannel]}
                strokeWidth="2"
                className="pointer-events-none"
              />
              
              {/* Control Points */}
              {currentCurve.map((point, index) => {
                const svgPos = toSVG(point[0], point[1]);
                return (
                  <circle
                    key={index}
                    cx={svgPos.x}
                    cy={svgPos.y}
                    r="6"
                    fill={channelColors[activeChannel]}
                    stroke="#fff"
                    strokeWidth="2"
                    className="cursor-move hover:r-8 transition-all"
                    onMouseDown={(e) => handleMouseDown(e, index)}
                    onDoubleClick={(e) => handlePointDoubleClick(e, index)}
                  />
                );
              })}
              
              {/* Axis labels */}
              <text x={padding + chartSize/2} y={padding + chartSize + 30} fill="#ccc" textAnchor="middle" fontSize="12">Input</text>
              <text x={15} y={padding + chartSize/2} fill="#ccc" textAnchor="middle" fontSize="12" transform={`rotate(-90, 15, ${padding + chartSize/2})`}>Output</text>
            </svg>
          </div>
          
          {/* Point Info */}
          <div className="mt-4 text-sm text-gray-400">
            <p>Click on curve to add points • Drag points to adjust • Double-click to remove (except endpoints)</p>
            <p className="mt-1">Current {activeChannel} curve: {currentCurve.map(p => `(${p[0]},${p[1]})`).join(' ')}</p>
          </div>
        </div>

        {/* Settings Panel */}
        <div className="w-80 bg-gray-800 p-4 rounded-lg">
          <h3 className="text-lg font-semibold mb-4">Settings</h3>
          
          {/* Blend Mode */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Blend Mode</label>
            <select 
              value={previewSettings.blendMode}
              onChange={(e) => setPreviewSettings(prev => ({...prev, blendMode: e.target.value}))}
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
            >
              <option value="normal">Normal</option>
              <option value="multiply">Multiply</option>
              <option value="screen">Screen</option>
              <option value="overlay">Overlay</option>
              <option value="soft_light">Soft Light</option>
              <option value="hard_light">Hard Light</option>
              <option value="color_dodge">Color Dodge</option>
              <option value="color_burn">Color Burn</option>
              <option value="darken">Darken</option>
              <option value="lighten">Lighten</option>
            </select>
          </div>

          {/* Opacity */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Opacity: {Math.round(previewSettings.opacity * 100)}%</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={previewSettings.opacity}
              onChange={(e) => setPreviewSettings(prev => ({...prev, opacity: parseFloat(e.target.value)}))}
              className="w-full"
            />
          </div>

          {/* Preserve Luminance */}
          <div className="mb-4">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={previewSettings.preserveLuminance}
                onChange={(e) => setPreviewSettings(prev => ({...prev, preserveLuminance: e.target.checked}))}
                className="mr-2"
              />
              <span className="text-sm">Preserve Luminance</span>
            </label>
          </div>

          {/* Export/Import */}
          <div className="space-y-2">
            <button className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded transition-colors">
              Export Curves
            </button>
            <button className="w-full bg-green-600 hover:bg-green-700 text-white py-2 px-4 rounded transition-colors">
              Import Curves
            </button>
            <button className="w-full bg-purple-600 hover:bg-purple-700 text-white py-2 px-4 rounded transition-colors">
              Apply to ComfyUI
            </button>
          </div>

          {/* Current Values Display */}
          <div className="mt-6 p-3 bg-gray-700 rounded">
            <h4 className="text-sm font-medium mb-2">Current Values</h4>
            <div className="text-xs space-y-1">
              {Object.entries(curves).map(([channel, points]) => (
                <div key={channel} className="flex justify-between">
                  <span className="capitalize" style={{color: channelColors[channel]}}>{channel}:</span>
                  <span>{points.length} points</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ApexCurveEditor;