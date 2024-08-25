# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container 
COPY . .

# Run script.py 
CMD ["python", "./script.py"]
