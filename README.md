# PharmaAlert Pro v4.0

Application de gestion d'officine (Maroc) — **PWA mono-fichier**, 100 % hors-ligne,
installable sur Android en application native (APK) ou via l'écran d'accueil.
Devise : Dirham (MAD). Interface français + argumentaires en darija.

## 📦 Contenu

| Fichier | Rôle |
|---|---|
| `index.html` | **L'application complète** (HTML + CSS + JS, SVG inline). Autonome. |
| `manifest.webmanifest` | Manifeste PWA (nom, icônes, mode plein écran). |
| `service-worker.js` | Cache hors-ligne (fonctionne sans internet). |
| `icons/` | Icônes de l'application (192, 512, maskable). |

## 🧩 Fonctionnalités (6 modules)

1. **Stock & Péremptions** — alertes 6 mois / 3 mois, valorisation en DH, méthode FEFO
   (lot expirant en premier), vente rapide **−1 boîte**, import/export **.xlsx / .csv**
   (compatible WinPharm / 3E / Officine).
2. **Retours grossistes** — détection de la fenêtre légale (3 à 6 mois), regroupement par
   répartiteur (Laprophan, Cooper Pharma, Sothema, Afric-Phar, Upsilon…), **bordereau A4
   officiel imprimable (PDF)** avec en-tête pharmacie + zones cachet/visa, envoi **WhatsApp**.
3. **Conseil comptoir & déstockage** — par famille thérapeutique, encart doré du produit
   prioritaire à déstocker (surstock / proche péremption), argumentaire **français + darija**.
4. **Réseau confraternel B2B** — carnet de pharmacies, bourse de surstock, radar **SOS
   rupture** WhatsApp, registre des ruptures nationales.
5. **Carnet de crédit (Kounach)** — créances par client, acomptes, relances WhatsApp
   diplomatiques, encours global.
6. **Profil de l'officine** — infos légales injectées dans les bordereaux + sauvegarde/restauration des données (JSON).

> 💾 Toutes les données restent **sur l'appareil** (localStorage). Pensez à exporter une
> sauvegarde régulière depuis **Profil → Sauvegarde (JSON)**.

## ▶️ Essayer immédiatement

Ouvrez simplement `index.html` dans Chrome. Pour activer le mode hors-ligne (service worker),
servez le dossier via HTTP :

```bash
cd Resto-Crepe
python3 -m http.server 8080
# puis ouvrir http://localhost:8080 dans le navigateur
```

Dans l'app : **Profil → « Charger données démo »** pour voir un exemple rempli.

## 📱 Installer sur Android

### Option A — Ajout à l'écran d'accueil (le plus simple, gratuit)
1. Hébergez le dossier sur une URL **HTTPS** (voir ci-dessous).
2. Ouvrez l'URL dans **Chrome Android**.
3. Menu ⋮ → **« Installer l'application »** / **« Ajouter à l'écran d'accueil »**.
   → L'app s'ouvre en plein écran, sans barre de navigateur, et marche hors-ligne.

### Option B — Générer un vrai APK

**PWABuilder (sans code, recommandé) :**
1. Publiez le dossier sur une URL HTTPS.
2. Allez sur <https://www.pwabuilder.com>, collez l'URL.
3. Onglet **Android → Generate Package** → téléchargez l'APK/AAB signé.
4. Transférez l'APK sur la tablette (Samsung Galaxy Tab A9…) et installez-le
   (autoriser « sources inconnues »).

**Bubblewrap (ligne de commande) :**
```bash
npm i -g @bubblewrap/cli
bubblewrap init --manifest https://VOTRE-URL/manifest.webmanifest
bubblewrap build        # produit app-release-signed.apk
```

### Hébergement HTTPS gratuit (au choix)
GitHub Pages, Netlify (glisser-déposer le dossier), Cloudflare Pages, Vercel.
> Si vous servez l'app dans un **sous-dossier**, les chemins sont déjà relatifs (`./`), donc
> aucune modification n'est nécessaire.

## 🔧 Notes techniques
- L'import **.csv** fonctionne toujours hors-ligne. L'import **.xlsx** charge la librairie
  SheetJS depuis un CDN à la volée (nécessite une connexion au moment de l'import) ; à défaut,
  exportez votre inventaire en `.csv` depuis votre logiciel d'officine.
- Numéros WhatsApp marocains nettoyés automatiquement (préfixe `+212`).
- Le bordereau A4 utilise l'impression native du navigateur → **« Enregistrer au format PDF »**
  pour archiver, ou imprimante directe.
- Après une mise à jour de `index.html`, incrémentez `CACHE_VERSION` dans
  `service-worker.js` pour forcer le rafraîchissement du cache.
