# PSIV2_tracking
En aquest respositori trobarem com fer el recompte de cotxes que passen per una via a partir d'un vídeo.

A continaució s'hi exposarà l'estructura de codi i l'explicació de cadascuna de les parts.

## Estructura
```
├───main.py
├───tracker.py
├───baseline.py
├───detect.py
```

## Baseline

El fitxer baseline.py conté el nostre primer approach per la tasca.

## Més enllà del Baseline

El fitxer main.py conté la millora de l'approach, on ja hi ha una diferenciació més explícita del detector de cotxes, el tracker i el comptador. En el mateix fitxer s'hi pot especificar quin és el tipus de detecció i recompte que es vol fer.

Possibles deteccions:
- Background substraction per morfologia
- YOLOv8

Possibles recomptes:
- Comptar quan el cotxe ha passat per complet
- Comptar quan el cotxe ha passat per la línia

![Segon mètode - una línia](https://i.imgur.com/nfb5V7Y.png)

- Comptar quan el cotxe ha passat per les dues línies.

![Tercer mètode - una línia](https://i.imgur.com/vggqwnw.png)



