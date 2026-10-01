# Start with a lightweight Python foundation
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install C++ compilers required for fastcluster and rapidfuzz
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy the dependencies file and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your code into the container
COPY . .

# Expose the port Streamlit uses for the web interface
EXPOSE 8501

# Command to launch the web interface when the container starts
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]