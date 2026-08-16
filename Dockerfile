# Multi-stage build: build Node apps and prepare Python model service
FROM ubuntu:22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y curl ca-certificates build-essential python3 python3-venv python3-pip git

# Install Node (18 LTS) and pnpm (optional)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Copy monorepo packages and install/build Next apps
WORKDIR /workspace
COPY package.json package-lock.json* ./
COPY packages/frontend ./packages/frontend
COPY packages/backend ./packages/backend
RUN if [ -f packages/frontend/package.json ]; then cd packages/frontend && npm ci && npm run build || true; fi
RUN if [ -f packages/backend/package.json ]; then cd packages/backend && npm ci && npm run build || true; fi

# Install Python deps for model service
COPY services/model ./services/model
WORKDIR /workspace/services/model
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip && \
    if [ -f requirements.txt ]; then /opt/venv/bin/pip install -r requirements.txt; fi

# Final image: minimal runtime
FROM ubuntu:22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y curl ca-certificates python3 python3-venv python3-pip nginx supervisor

# Install Node runtime
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && apt-get install -y nodejs

WORKDIR /app

# Copy built Next apps
COPY --from=builder /workspace/packages/frontend/.next /app/frontend/.next
COPY --from=builder /workspace/packages/frontend/package.json /app/frontend/package.json
COPY --from=builder /workspace/packages/frontend/public /app/frontend/public
COPY --from=builder /workspace/packages/backend/.next /app/backend/.next
COPY --from=builder /workspace/packages/backend/package.json /app/backend/package.json

# Copy model service and its venv
COPY --from=builder /workspace/services/model /app/services/model
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
ENV NODE_ENV=production

# Supervisor config to run model (uvicorn) and both Next apps (next start)
RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 3000 3001 8000

HEALTHCHECK --interval=20s --timeout=3s --start-period=10s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
