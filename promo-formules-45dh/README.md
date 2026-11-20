# Promo « 2 formules à 45 DH » — du 22 au 29 novembre

Publicité verticale (9:16) pour Instagram, TikTok et Facebook, 30 s.

| Fichier | Usage |
|---|---|
| `resto_crepes_formules_45dh_4K.mp4` | Master 4K (2160×3840, 30 i/s, H.264, AAC 48 kHz, -14 LUFS) |
| `resto_crepes_formules_45dh_1080p.mp4` | Version légère 1080×1920 pour l'upload direct |
| `miniature.jpg` | Image de couverture (1080×1920) |

## Montage

| Temps | Séquence |
|---|---|
| 0:00 – 0:04 | Accroche : arrivée à table, « Chnou katakoul ? », « Resto Crêpes dayrin promo harba ! » |
| 0:04 – 0:08 | Montage rapide des préparations + « 2 formules · 45 DH · 1 semaine seulement » |
| 0:08 – 0:15 | Formule pizza : pizza 4 fromages + pancake + boisson offerte → carte prix 45 DH |
| 0:15 – 0:22 | Formule tacos : tacos poulet + frites + crêpe Nutella + boisson offerte → carte prix 45 DH |
| 0:22 – 0:24 | « Fin had chi ? » « F Resto Crêpes ! » |
| 0:24 – 0:30 | Écran de fin (2 formules, dates, Rabat · Salé · Témara) + « On vous attend ! » |

Son : dialogue d'origine (IMG_0143) nettoyé, musique et effets 100 % originaux (synthétisés, sans droits tiers).
Image : plans agrandis en 4K par super-résolution (ESPCN ×2), étalonnage, graphismes rendus nativement en 4K.

## Reconstruire

Les rushes (release GitHub `Crepes`) doivent être dans `$PROMO_WORK/videos` et `$PROMO_WORK/photos`,
les polices (Anton, Montserrat, Kaushan Script — Google Fonts) dans `$PROMO_WORK/fonts`,
le modèle `ESPCN_x2.pb` dans `$PROMO_WORK/models`, et les scripts dans `$PROMO_WORK/work`.

```bash
python3 work/prep.py                      # plans -> cache 4K (super-résolution)
python3 work/voice45.py videos/IMG_0143.MOV aud/voice.wav
python3 work/music45.py aud/music.wav aud/fx.wav
python3 work/mix45.py aud/voice.wav aud/music.wav aud/fx.wav aud/mix.wav
python3 work/build45.py out_4k.mp4 --audio aud/final.wav   # final.wav = mix.wav normalisé à -14 LUFS
```

Le minutage des plans est dans `scripts/edit.py`, les sous-titres, puces et cartes prix en tête de `scripts/build45.py`.
