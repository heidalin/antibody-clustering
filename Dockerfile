FROM public.ecr.aws/docker/library/python:3.10-slim

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    while ! pip install --default-timeout=2000 --retries=50 -r requirements.txt; do \
        echo "Network dropped! Resuming download..."; \
        sleep 3; \
    done

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
