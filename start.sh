#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# DeepGuard — Start everything
# Usage: ./start.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

info()    { echo -e "${GREEN}[deepguard]${NC} $1"; }
warning() { echo -e "${YELLOW}[deepguard]${NC} $1"; }
error()   { echo -e "${RED}[deepguard]${NC} $1"; exit 1; }

# ── Check Python version ──────────────────────────────────────────────────────
if command -v python3.11 &>/dev/null; then
  PYTHON=python3.11
elif command -v python3 &>/dev/null; then
  PYTHON=python3
  warning "Python 3.11 recommended for full compatibility"
else
  error "Python 3 not found. Install Python 3.11."
fi

# ── Check .env ────────────────────────────────────────────────────────────────
if [ ! -f "$ROOT/.env" ]; then
  warning ".env not found — copying from .env.example"
  cp "$ROOT/.env.example" "$ROOT/.env"
  warning "Please fill in your values in .env before running in production"
fi

# ── Python virtual environment ────────────────────────────────────────────────
if [ ! -d "$ROOT/venv" ]; then
  info "Creating Python virtual environment..."
  $PYTHON -m venv "$ROOT/venv"
fi

source "$ROOT/venv/bin/activate"

info "Installing Python dependencies..."
pip install -q -r "$ROOT/requirements.txt"

# ── Start FastAPI backend ─────────────────────────────────────────────────────
info "Starting FastAPI backend on http://localhost:8000 ..."
cd "$ROOT"
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
info "Backend PID: $BACKEND_PID"

# ── Install and start frontend ────────────────────────────────────────────────
cd "$ROOT/frontend"
if [ ! -d "node_modules" ]; then
  info "Installing frontend dependencies..."
  npm install
fi

info "Starting React frontend on http://localhost:3000 ..."
npm run dev &
FRONTEND_PID=$!
info "Frontend PID: $FRONTEND_PID"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  DeepGuard is running!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Frontend  →  ${YELLOW}http://localhost:3000${NC}"
echo -e "  Backend   →  ${YELLOW}http://localhost:8000${NC}"
echo -e "  API docs  →  ${YELLOW}http://localhost:8000/docs${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Press Ctrl+C to stop both servers."

# ── Wait and cleanup ──────────────────────────────────────────────────────────
trap "info 'Stopping...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
