/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  images: {
    unoptimized: true,
  },
  // Trailing slashes for GitHub Pages compatibility
  trailingSlash: true,
};

module.exports = nextConfig;

