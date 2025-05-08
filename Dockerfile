FROM python:3.9-slim

WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for data and uploads
RUN mkdir -p /app/uploads /app/data

# Set permissions for the directories
RUN chmod 777 /app/uploads /app/data

# Set environment variables
ENV PORT=9099
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Copy application code (after creating directories)
COPY . .

# Volumes for persistent data
VOLUME ["/app/uploads", "/app/data"]

# Expose port
EXPOSE 9099

# Run the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:9099", "--timeout", "600", "app:app"] 