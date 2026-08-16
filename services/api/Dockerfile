# syntax=docker/dockerfile:1

ARG PNPM_VERSION=9.15.4

FROM node:20-alpine AS base
RUN apk add --no-cache libc6-compat wget
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
RUN corepack enable && corepack prepare pnpm@${PNPM_VERSION} --activate

FROM base AS deps
WORKDIR /app
COPY package.json pnpm-workspace.yaml pnpm-lock.yaml .npmrc ./
COPY services/api/package.json ./services/api/
RUN pnpm install --frozen-lockfile --filter goemotions-api...

FROM base AS development
WORKDIR /app
COPY package.json pnpm-workspace.yaml pnpm-lock.yaml .npmrc ./
COPY services/api/package.json ./services/api/
RUN pnpm install --filter goemotions-api
COPY services/api ./services/api
EXPOSE 4000
CMD ["pnpm", "--filter", "goemotions-api", "start:dev"]

FROM base AS builder
WORKDIR /app
COPY package.json pnpm-workspace.yaml pnpm-lock.yaml .npmrc ./
COPY services/api ./services/api
RUN pnpm install --frozen-lockfile --filter goemotions-api
RUN pnpm --filter goemotions-api build
RUN pnpm deploy --filter goemotions-api --prod /prod/api

FROM node:20-alpine AS production
RUN apk add --no-cache wget
WORKDIR /app
ENV NODE_ENV=production \
    PORT=4000 \
    MODEL_SERVICE_URL=http://model:8000

COPY --from=builder /prod/api ./

EXPOSE 4000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD wget -qO- http://localhost:4000/api/health || exit 1
CMD ["node", "dist/main.js"]
