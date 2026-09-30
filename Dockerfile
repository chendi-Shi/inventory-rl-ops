FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY models ./models
RUN pip install --no-cache-dir '.[api]'
ENV MODEL_DIR=/models
ENV PORTFOLIO_MODEL_DIR=/models
ENV POLICY_SEARCH_MODEL_DIR=/app/models/wi1-policy-search
ENV CONTEXT_MODEL_DIR=/app/models/tx3-context
EXPOSE 8000
CMD ["uvicorn", "inventory_rl.api:app", "--host", "0.0.0.0", "--port", "8000"]
