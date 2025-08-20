'use client'

import { useState, useEffect } from 'react'
import { Play, Pause, RotateCcw } from 'lucide-react'

export function InternalsPage() {
  const [isAnimating, setIsAnimating] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)

  const steps = [
    {
      title: 'Video Input',
      description: 'Raw video frame captured from camera',
      color: 'bg-blue-500',
      position: { x: 50, y: 50 }
    },
    {
      title: 'Face Detection',
      description: 'RetinaFace detects facial landmarks',
      color: 'bg-green-500',
      position: { x: 200, y: 50 }
    },
    {
      title: 'Feature Extraction',
      description: 'CNN extracts facial features',
      color: 'bg-purple-500',
      position: { x: 350, y: 50 }
    },
    {
      title: 'AU Classification',
      description: 'Action Units are classified',
      color: 'bg-orange-500',
      position: { x: 500, y: 50 }
    },
    {
      title: 'Emotion Analysis',
      description: 'Emotions are derived from AUs',
      color: 'bg-red-500',
      position: { x: 350, y: 200 }
    },
    {
      title: 'Gaze Estimation',
      description: 'Eye gaze direction calculated',
      color: 'bg-cyan-500',
      position: { x: 200, y: 200 }
    },
    {
      title: 'Output Visualization',
      description: 'Results rendered on interface',
      color: 'bg-yellow-500',
      position: { x: 50, y: 200 }
    }
  ]

  useEffect(() => {
    if (isAnimating) {
      const interval = setInterval(() => {
        setCurrentStep(prev => (prev + 1) % steps.length)
      }, 1500)
      return () => clearInterval(interval)
    }
  }, [isAnimating, steps.length])

  const handlePlay = () => {
    setIsAnimating(!isAnimating)
  }

  const handleReset = () => {
    setIsAnimating(false)
    setCurrentStep(0)
  }

  return (
    <div className="max-w-7xl mx-auto p-5">
      <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-lg">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-3 text-gray-900 dark:text-white">
            🔬 OpenFace Pipeline Internals
          </h1>
          <p className="text-gray-600 dark:text-gray-400 text-lg">
            Understanding the data flow through the facial analysis pipeline
          </p>
        </div>

        {/* Animation Controls */}
        <div className="flex justify-center gap-4 mb-8">
          <button
            onClick={handlePlay}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
          >
            {isAnimating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isAnimating ? 'Pause Animation' : 'Play Animation'}
          </button>
          <button
            onClick={handleReset}
            className="flex items-center gap-2 px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>
        </div>

        {/* Pipeline Visualization */}
        <div className="relative bg-gray-50 dark:bg-gray-800 rounded-lg p-8 mb-8 min-h-[400px]">
          <svg
            className="absolute inset-0 w-full h-full"
            viewBox="0 0 600 300"
          >
            {/* Connection lines */}
            <defs>
              <marker
                id="arrowhead"
                markerWidth="10"
                markerHeight="7"
                refX="9"
                refY="3.5"
                orient="auto"
              >
                <polygon
                  points="0 0, 10 3.5, 0 7"
                  fill="currentColor"
                  className="text-gray-400 dark:text-gray-500"
                />
              </marker>
            </defs>
            
            {/* Flow lines */}
            <g className="text-gray-400 dark:text-gray-500">
              <line x1="100" y1="75" x2="180" y2="75" stroke="currentColor" strokeWidth="2" markerEnd="url(#arrowhead)" />
              <line x1="250" y1="75" x2="330" y2="75" stroke="currentColor" strokeWidth="2" markerEnd="url(#arrowhead)" />
              <line x1="400" y1="75" x2="480" y2="75" stroke="currentColor" strokeWidth="2" markerEnd="url(#arrowhead)" />
              <line x1="500" y1="100" x2="400" y2="180" stroke="currentColor" strokeWidth="2" markerEnd="url(#arrowhead)" />
              <line x1="330" y1="225" x2="250" y2="225" stroke="currentColor" strokeWidth="2" markerEnd="url(#arrowhead)" />
              <line x1="180" y1="225" x2="100" y2="225" stroke="currentColor" strokeWidth="2" markerEnd="url(#arrowhead)" />
            </g>
          </svg>

          {/* Process nodes */}
          {steps.map((step, index) => (
            <div
              key={index}
              className={`absolute transform -translate-x-1/2 -translate-y-1/2 w-24 h-24 rounded-full ${step.color} flex items-center justify-center transition-all duration-300 ${
                currentStep === index
                  ? 'scale-125 ring-4 ring-white dark:ring-gray-800 shadow-lg'
                  : 'scale-100'
              } ${
                isAnimating && currentStep === index
                  ? 'animate-pulse'
                  : ''
              }`}
              style={{
                left: `${step.position.x}px`,
                top: `${step.position.y}px`
              }}
            >
              <span className="text-white font-bold text-xs text-center leading-tight">
                {index + 1}
              </span>
            </div>
          ))}
        </div>

        {/* Current Step Details */}
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
            Step {currentStep + 1}: {steps[currentStep].title}
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            {steps[currentStep].description}
          </p>
        </div>

        {/* Technical Details */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mt-8">
          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold mb-2 text-gray-900 dark:text-white">Architecture</h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• ResNet-50 Backbone</li>
              <li>• Multi-task Learning</li>
              <li>• Attention Mechanisms</li>
            </ul>
          </div>
          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold mb-2 text-gray-900 dark:text-white">Features</h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• 17 Action Units</li>
              <li>• 7 Basic Emotions</li>
              <li>• 3D Gaze Vector</li>
            </ul>
          </div>
          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-semibold mb-2 text-gray-900 dark:text-white">Performance</h4>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>• Real-time Processing</li>
              <li>• 30+ FPS Capability</li>
              <li>• GPU Accelerated</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
