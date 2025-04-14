# Use the official PyTorch base image
FROM pytorch/pytorch:latest

# Set the working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy our application code and models into the container
COPY inference.py /app/inference.py
COPY models/ /app/models/

# Expose port 8080 for Flask
EXPOSE 8080

# Run the Flask inference API
CMD ["python", "inference.py"]
