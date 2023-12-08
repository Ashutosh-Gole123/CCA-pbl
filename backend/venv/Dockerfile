# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /usr/src/backend

# Copy the requirements file into the container at /usr/src/app
COPY requirements.txt ./

# Install OpenCV dependencies
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx 


RUN /usr/local/bin/python -m pip install --upgrade pip

RUN pip install opencv-python-headless

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local code to the container
COPY . .

EXPOSE 5000

# Specify the command to run on container start
CMD ["python", "app.py"]