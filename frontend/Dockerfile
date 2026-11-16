FROM node:22-alpine AS build
WORKDIR /app
ARG VITE_API_URL=http://localhost:8000
ARG VITE_WEBCAM_INTERVAL_MS=300
ENV VITE_API_URL=$VITE_API_URL
ENV VITE_WEBCAM_INTERVAL_MS=$VITE_WEBCAM_INTERVAL_MS
COPY frontend/package*.json ./
RUN npm install --no-audit --no-fund
COPY frontend ./
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
HEALTHCHECK --interval=20s --timeout=3s --retries=3 CMD wget -qO- http://localhost/ >/dev/null || exit 1
EXPOSE 80
