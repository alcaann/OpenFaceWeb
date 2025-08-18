# OpenFace-3.0 Web Client

A modern Next.js web application for real-time facial analysis using the OpenFace-3.0 backend API.

## Features

- 🎥 Real-time webcam capture
- 🔍 Live facial analysis with WebSocket connection
- 📊 Display of facial landmarks, emotions, gaze direction, and Action Units
- 🎨 Visual overlay on video feed
- 📱 Responsive design for desktop and mobile
- ⚙️ Configurable frame rate and video quality
- 📸 Screenshot capture functionality

## Prerequisites

- Node.js 18.x or later
- NPM or Yarn package manager
- OpenFace-3.0 backend API running (default: http://localhost:5000)
- Webcam access

## Installation

1. Navigate to the project directory:
```bash
cd openface-web
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open your browser and navigate to:
```
http://localhost:3000
```

## Usage

1. **Start the Backend**: Ensure the OpenFace-3.0 API server is running on port 5000
2. **Connect**: Click "Connect" to establish WebSocket connection with the backend
3. **Start Camera**: Click "Start Camera" to access your webcam
4. **Begin Analysis**: Click "Start Analysis" to begin real-time facial analysis
5. **Adjust Settings**: Use the sliders to modify frame rate and video quality
6. **View Results**: Analysis results will appear below the video feeds

## Configuration

### Server URL
Change the server URL in the settings if your backend is running on a different host/port.

### Frame Rate
Adjust how many frames per second are sent for analysis (1-30 FPS).

### Video Quality
Control the JPEG compression quality for transmitted frames (0.1-1.0).

## Project Structure

```
openface-web/
├── src/
│   └── app/
│       ├── globals.css          # Global styles
│       ├── layout.tsx          # Root layout component
│       └── page.tsx            # Main application component
├── package.json                # Dependencies and scripts
├── tsconfig.json              # TypeScript configuration
├── next.config.js             # Next.js configuration
└── README.md                  # This file
```

## Key Components

### Main Features
- **WebSocket Communication**: Real-time bidirectional communication with backend
- **Video Capture**: Browser-based webcam access with MediaDevices API
- **Canvas Overlay**: Real-time rendering of analysis results over video
- **State Management**: React hooks for component state and lifecycle management

### Analysis Overlays
- **Bounding Boxes**: Green rectangles around detected faces
- **Landmarks**: Red dots for facial landmark points
- **Emotions**: Text display of emotion classification
- **Gaze Direction**: Orange arrows showing gaze vector
- **Action Units**: Display of active facial action units

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

### Building for Production

```bash
npm run build
npm run start
```

## Browser Compatibility

- Chrome 80+
- Firefox 75+
- Safari 14+
- Edge 80+

Requires modern browser support for:
- WebRTC/MediaDevices API
- WebSocket
- Canvas API
- ES6+ features

## Troubleshooting

### Camera Access Issues
- Ensure HTTPS is used for production (required for camera access)
- Check browser permissions for camera access
- Verify no other applications are using the camera

### Connection Issues
- Verify the backend API is running
- Check the server URL configuration
- Ensure CORS is properly configured on the backend

### Performance Issues
- Reduce frame rate for better performance
- Lower video quality to reduce bandwidth
- Close other resource-intensive applications

## License

This project is part of the OpenFace-3.0 suite. Please refer to the main project license.
