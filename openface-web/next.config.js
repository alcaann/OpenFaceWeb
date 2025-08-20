/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  // Disabled SWC plugin for LinguiJS due to segmentation faults
  // Using Babel plugin instead
}

module.exports = nextConfig
