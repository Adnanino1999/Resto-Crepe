/* PharmaAlert Pro - Service Worker
   Cache applicatif pour fonctionnement 100% hors-ligne.
   Incrémenter CACHE_VERSION à chaque mise à jour de l'app. */
const CACHE_VERSION = 'pharmaalert-v5.4.0';
const ASSETS = [
  './',
  './index.html',
  './manifest.webmanifest',
  './assets/pharmacie-bg.jpg',
  './icons/icon-192.png',
  './icons/icon-512.png',
  './icons/icon-maskable-512.png'
];

self.addEventListener('message', (e) => {
  if (e.data === 'skipWaiting') self.skipWaiting();
});

self.addEventListener('install', (e) => {
  self.skipWaiting();
  e.waitUntil(caches.open(CACHE_VERSION).then((c) => c.addAll(ASSETS)).catch(() => {}));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_VERSION).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  // Ne pas intercepter les ressources tierces (ex: CDN SheetJS) : réseau direct.
  if (url.origin !== self.location.origin) return;

  // Navigation : réseau d'abord, repli sur le cache (app-shell).
  if (req.mode === 'navigate') {
    e.respondWith(
      fetch(req).catch(() => caches.match('./index.html'))
    );
    return;
  }

  // Autres ressources same-origin : cache d'abord, puis réseau (et mise en cache).
  e.respondWith(
    caches.match(req).then((cached) =>
      cached || fetch(req).then((res) => {
        const copy = res.clone();
        caches.open(CACHE_VERSION).then((c) => c.put(req, copy)).catch(() => {});
        return res;
      }).catch(() => cached)
    )
  );
});
