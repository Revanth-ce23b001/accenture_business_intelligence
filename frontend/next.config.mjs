/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The Python API. The browser never talks to it directly: every call
  // goes through app/api/casefile/[...path], which mints the signed
  // persona header server-side. See lib/token.ts for why.
  env: {
    CASEFILE_API: process.env.CASEFILE_API ?? "http://127.0.0.1:8000",
  },
};
export default nextConfig;
