# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /flask_app

# Copy requirement list first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all local project files into the container
COPY . .

# Expose Flask's default port
EXPOSE 5000

# Run the Flask app using Gunicorn (or flask run)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "flask_app:app"]