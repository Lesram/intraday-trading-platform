# 🐳 Dockerfile for Intraday Trading Platform
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage  
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY backend/ ./backend/
COPY .env.example .env

# Create non-root user for security
RUN addgroup --system --gid 1001 trading && \
    adduser --system --uid 1001 --gid 1001 trading && \
    chown -R trading:trading /app

# Switch to non-root user
USER trading

# Add local Python packages to PATH
ENV PATH=/root/.local/bin:$PATH

# Environment variables
ENV PYTHONPATH=/app/backend
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8002/api/health || exit 1

# Expose port
EXPOSE 8002

# Add labels for better container management
LABEL maintainer="Intraday Trading Platform Team"
LABEL version="2.0.0"
LABEL description="Layer 2 Trading Platform with FastAPI, Prometheus metrics, and authentication"

# Start command with proper signal handling
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
