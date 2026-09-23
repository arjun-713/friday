/** @type {import('next').NextConfig} */
const config = {
  // Keep Fast Refresh and production verification from overwriting each other.
  distDir: process.env.NODE_ENV === "development" ? ".next-dev" : ".next",
};

export default config;
