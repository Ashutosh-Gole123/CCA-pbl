# Use an official Node runtime as a parent image
FROM node

# Set the working directory in the container
WORKDIR /usr/src/frontend

# Copy package.json and package-lock.json to the working directory
COPY package*.json ./

# Install app dependencies
RUN npm install

# Copy the local code to the container
COPY . .

# Build the React app
RUN npm run build

EXPOSE 5173

# Specify the command to run on container start
CMD ["npm", "run", "dev"]