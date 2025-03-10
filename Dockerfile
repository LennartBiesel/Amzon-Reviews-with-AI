# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Create a virtual environment, upgrade pip, and install requirements
RUN python -m venv venv && \
    venv/bin/pip install --upgrade pip && \
    venv/bin/pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Run the application using the Python executable from the virtual environment
CMD ["venv/bin/python", "app.py"]
