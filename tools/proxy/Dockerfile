FROM node:20-bookworm-slim

WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev
COPY . .
USER node
CMD ["npm", "start"]
