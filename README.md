# Cel projektu:
Celem projektu jest estymacja kierunku źródła światła na podstawie analizy geometrii cieni w obrazie.

## Wymagania
- Python 3.10+ 
- Zależności z `requirements.txt` 
- Dane: folder z plikami lub sam plik z cieniami (obrazy/GIF-y/wideo)

## Przygotowanie środowiska

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Uruchomienie
Domyślnie aplikacja otworzy okno wyboru źródła (pojedynczy plik lub folder).
Obsługiwane są obrazy, GIF-y i wideo (bez dźwięku). Pauza działa tym samym przyciskiem dla GIF i wideo.

```powershell
python main.py
```

## Dane

Wykorzystano zbiór SBU Shadow Dataset zawierający obrazy z cieniami.
https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html

Do testów danych cieni możesz też użyć zbioru VIDIT.
https://github.com/majedelhelou/VIDIT

## Ewaluacja (box-shadow.mp4)

Skrypt zaklada, ze w pierwszej klatce kierunek cienia to ok. 30-40 deg, a potem kierunek obraca sie przeciwnie do ruchu wskazowek zegara i wraca do punktu startu.
Mozesz dostosowac `--start-angle` i `--rotation-deg`.

```powershell
python evaluate_box_shadow.py --video data\box-shadow.mp4 --start-angle 35 --rotation-deg 360
```

Opcjonalnie zapis per-klatka do CSV:

```powershell
python evaluate_box_shadow.py --video data\box-shadow.mp4 --start-angle 35 --rotation-deg 360 --save-csv out_eval.csv
```

## Wykresy z CSV (ewaluacja)

Mini program rysuje wykresy bledu i confidence z CSV:

```powershell
python plot_eval_csv.py --csv out_eval.csv
```

Mozesz pominac `--csv` i wybrac plik z okna:

```powershell
python plot_eval_csv.py
```

Opcjonalnie zapisz obraz:

```powershell
python plot_eval_csv.py --csv out_eval.csv --save out_eval.png
```
