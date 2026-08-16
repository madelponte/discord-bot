FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .

# The bot only reads files and opens outbound network connections — it never
# needs root or writable paths, so drop privileges for the runtime.
USER nobody

CMD ["python", "-u", "bot.py"]
