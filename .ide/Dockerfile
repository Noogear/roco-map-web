FROM docker.cnb.cool/wss/ai/mlsharp:latest

ENV PREBUILT_ENV_ROOT=/opt/game-map-web-env \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/game-map-web-env/.venv/bin:${PATH}"

WORKDIR ${PREBUILT_ENV_ROOT}

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --extra ai --no-install-project \
    && .venv/bin/python -c "import torch, kornia; print(f'prebuilt torch={torch.__version__}'); print(f'prebuilt kornia={kornia.__version__}'); print(f'cuda_available={torch.cuda.is_available()}')"

WORKDIR /workspace
