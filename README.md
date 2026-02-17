# Cel projektu:
Celem projektu jest estymacja kierunku źródła światła oraz rozróżnienie światła naturalnego i sztucznego na podstawie analizy geometrii cieni w obrazie.

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
https://github.com/majedelhelou/VIDIT?utm_source=chatgpt.com