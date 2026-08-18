FROM node:22-bookworm-slim AS build

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend ./
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL
RUN npm run build

FROM node:22-bookworm-slim

WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app ./

EXPOSE 3000
CMD ["npm", "run", "start"]
