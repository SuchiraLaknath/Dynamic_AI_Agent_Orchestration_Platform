FROM node:22-slim

WORKDIR /srv/frontend

COPY frontend/package.json /srv/frontend/package.json
RUN npm install

COPY frontend /srv/frontend

EXPOSE 5173
CMD ["npm", "run", "dev"]
