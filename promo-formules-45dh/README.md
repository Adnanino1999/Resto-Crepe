# Promo « 2 formules à 45 DH » — du 25 novembre au 1er décembre 2026 (inclus)

Publicité verticale (9:16) pour Instagram, TikTok et Facebook, 49 s, ton humoristique.

| Fichier | Usage |
|---|---|
| `resto_crepes_formules_45dh_4K.mp4` | Version 4K (2160×3840, 30 i/s, HEVC/H.265, AAC 48 kHz, -14 LUFS) |
| `resto_crepes_formules_45dh_1080p.mp4` | 1080×1920 H.264 : lecture partout, upload direct |
| `miniature.jpg` | Image de couverture (1080×1920) |

## Montage

| Temps | Séquence |
|---|---|
| 0:00 – 0:04 | Accroche : arrivée à table, « Chnou katakoul ? », « Resto Crêpes dayrin promo harba ! » |
| 0:04 – 0:08 | Montage rapide des préparations + « 2 formules · 45 DH · 1 semaine seulement » |
| 0:08 – 0:15 | Formule pizza : pizza 4 fromages + pancake + boisson offerte → carte prix 45 DH |
| 0:15 – 0:22 | Formule tacos : tacos poulet + frites + crêpe Nutella + boisson offerte → carte prix 45 DH |
| 0:22 – 0:24 | « Fin had chi ? » « F Resto Crêpes ! » |
| 0:24 – 0:25 | Transition « من بعد شي سوايع... » (avance rapide + réveil) |
| 0:25 – 0:42 | Le retour : il mange encore (formule tacos), « Mazal ba9i katakoul ?! », il s'énerve → arrêt sur image comique 😤, il bouscule la caméra |
| 0:42 – 0:47 | Écran de fin : les 2 formules, dates (25 nov. → 1er déc.), puis 📞 06 60 24 14 75 · 🛵 Livraison Rabat • Témara · 📍 Hay El Fath – Rabat |
| 0:47 – 0:49 | « On vous attend ! » + numéro |

Pas de sous-titres (public marocain, dialogue en darija).
Son : dialogues d'origine (IMG_0143 et IMG_0177) nettoyés ; musique originale v2 (afro-house aux couleurs marocaines,
darbouka, oud, mode Hijaz) et effets 100 % synthétisés, sans droits tiers.
Image : plans agrandis en 4K par super-résolution (ESPCN ×2), étalonnage, graphismes rendus nativement en 4K.

## Reconstruire

Les rushes (release GitHub `Crepes`) doivent être dans `$PROMO_WORK/videos` et `$PROMO_WORK/photos`,
les polices (Anton, Montserrat, Kaushan Script, Lalezar — Google Fonts) dans `$PROMO_WORK/fonts`,
le modèle `ESPCN_x2.pb` dans `$PROMO_WORK/models`, et les scripts dans `$PROMO_WORK/work`.

```bash
python3 work/prep.py                      # plans -> cache 4K (super-résolution)
python3 work/voice45.py videos aud/voice.wav
python3 work/music_hijaz.py aud/music.wav aud/fx.wav
python3 work/mix45.py aud/voice.wav aud/music.wav aud/fx.wav aud/mix.wav
python3 work/build45.py out_4k.mp4 --audio aud/final.wav   # final.wav = mix.wav normalisé à -14 LUFS
```

Le minutage des plans est dans `scripts/edit.py` ; puces, cartes prix, écran de fin et coordonnées en tête de `scripts/build45.py`.
