#!/bin/bash

# OpenFace Web Client Setup Script
# This script sets up the Next.js web client for OpenFace-3.0

set -e

echo "🚀 Setting up OpenFace Web Client..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18.x or later."
    echo "Visit: https://nodejs.org/"
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js version 18 or later is required. Current version: $(node -v)"
    exit 1
fi

echo "✅ Node.js $(node -v) detected"

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm."
    exit 1
fi

echo "✅ npm $(npm -v) detected"

# Navigate to the web client directory
cd "$(dirname "$0")"

echo "📦 Installing dependencies..."
npm install

echo "🔧 Building the application..."
npm run build

echo "✅ Setup complete!"
echo ""
echo "🎯 To start the development server:"
echo "   npm run dev"
echo ""
echo "🌐 To start the production server:"
echo "   npm run start"
echo ""
echo "📝 Make sure the OpenFace-3.0 backend API is running on port 5000"
echo "   Then open http://localhost:3000 in your browser"
