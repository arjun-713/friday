FROM node:22-bookworm-slim AS build

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci

COPY frontend ./
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL
# One layer: build, then drop everything `next start` doesn't need.
# .next/cache alone is ~150MB of regenerable build cache; devDependencies
# (typescript, playwright, @types) are unused at runtime.
# Never let a host or earlier build's generated route output leak into the
# production image; the context intentionally excludes these directories.
RUN rm -rf .next .next-dev \
    && npm run build \
    && rm -rf .next/cache \
    && npm prune --omit=dev

FROM node:22-bookworm-slim

WORKDIR /app
ENV NODE_ENV=production
# Copy only runtime inputs: previously the whole /app (sources, tests,
# .next-dev, build cache) shipped into the final image (~1.27GB).
COPY --from=build /app/package.json /app/package-lock.json ./
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/.next ./.next
COPY --from=build /app/public ./public
COPY --from=build /app/next.config.mjs ./next.config.mjs

EXPOSE 3000
CMD ["npm", "run", "start"]
