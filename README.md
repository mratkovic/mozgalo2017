# Korisnička dokumentacija

## Instalacija

Projektno rješenje pisano je u programskom jeziku Python 3 koristeći čitav niz pomoćnih biblioteka (Tensorflow, scikit-learn, scikit-image…).

U direktoriju s izvornim kodom nalazi se datoteka `requirements.txt` koja sadrži popis potrebnih biblioteka.
Preporuka je da koristite virtual environment za njihovu instalaciju.

Instalacija virtualenv:
```sh
pip3 install virtualenv
```


Stvaranje novog virtualnog okruženja
```sh
virtualenv -p /usr/bin/python3 mozgalo
```


Aktivacija stvorenog virtualnog okruženja
```sh
source ./mozgalo/bin/activate
```


Instalacija svih potrebnih paketa:
```sh
pip3 install -r requirements.txt

```  
-----
## Skupovi podataka

Korišteno je nekoliko javno dostupnih skupova podataka za evaluaciju rješenja:
 - [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)
 - [stl10](http://cs.stanford.edu/~acoates/stl10/)
 - [Cats_vs_dogs](https://www.kaggle.com/c/dogs-vs-cats)
 
Kako bi pojednostavnili skidanje i pohranu svih skupova podataka potrebnih za reproduciranje rezultata pripremili smo jednu arhivu koja sadrži sve potrebno na sljedećoj [poveznici](https://www.dropbox.com/s/7ui2a58md0zvhep/dataset.tar.gz?dl=0). Dobivenu arhivu raspakirati u korijenski direktorij projekta.
```sh
wget wget -O dataset.tar.gz 'https://www.dropbox.com/s/7ui2a58md0zvhep/dataset.tar.gz?dl=0'
tar xvf dataset.tar.gz
```
-----
## Pokretanje
U direktoriju `notebooks` nalazi se jupyter bilježnica s cjelovitim rješenjem.
```sh
cd notebooks
jupyter notebook final_report.ipynb
```
Odlučili smo se za korištenje python bilježnice radi jednostavnosti korištenja, pogodnosti za demonstraciju rezultata, laganog uređivanja i eksperimentiranja nad rezultatima. 

U samoj bilježnici slijedno je prikazan postupak:
- učitavanje podataka
- izlučivanje značajki
- odabir optimalne dimenzije značajki i redukcija dimenzionalnosti 
- odabir optimalnog broja grupa i samo grupiranje
- prikaz rezultata
- provjera rezultata nad označenim skupovima podataka 


----
## Struktura direktorija
- `dataset` - direktorij sa skupovima podataka
- `notebooks` - korištene Jupyter bilježnice
- `models` - prethodno trenirani TensorFlow modeli korišteni za ekstrakciju značajki
- `report_imgs` - slike korištene u dokumentaciji
- `clusters` - dobiveni rezultati grupiranja
- `src` - direktorij s izvornim kodom
- `src/modules/external` - git submodule iz paketa `slim` i `keras` za korištenje gotovih prethodno treniranih mreža 

____
Priloženo programsko rješenje je isprobano pod Linux okruženjem (Ubuntu 16.04 i Arch Linux). 
Ako naiđete na ikakav problem prilikom pokretanja, molimo Vas da se javite nekom od članova tima kako bi otklonili moguće poteškoće.
