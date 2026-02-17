import tkinter as tk
from tkinter import ttk, filedialog


class SourceDialogMixin:
    def _prompt_source(self):
        result = {"mode": None, "path": None, "folder": None}

        dlg = tk.Toplevel(self)
        dlg.title("Wybierz zrodlo")
        dlg.resizable(False, False)
        dlg.grab_set()

        ttk.Label(dlg, text="Wybierz folder lub pojedynczy obraz:", padding=12).grid(row=0, column=0, columnspan=3)

        def pick_file():
            path = filedialog.askopenfilename(
                parent=dlg,
                title="Wybierz obraz lub GIF",
                filetypes=[
                    ("Obrazy", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.gif"),
                    ("Wideo", "*.mp4;*.avi;*.mov;*.mkv;*.webm"),
                    ("Wszystkie pliki", "*.*"),
                ],
            )
            if path:
                result["mode"] = "file"
                result["path"] = path
            dlg.destroy()

        def pick_folder():
            folder = filedialog.askdirectory(parent=dlg, title="Wybierz folder z obrazami")
            if folder:
                result["mode"] = "folder"
                result["folder"] = folder
            dlg.destroy()

        def cancel():
            dlg.destroy()

        ttk.Button(dlg, text="Plik", command=pick_file).grid(row=1, column=0, padx=8, pady=10)
        ttk.Button(dlg, text="Folder", command=pick_folder).grid(row=1, column=1, padx=8, pady=10)
        ttk.Button(dlg, text="Anuluj", command=cancel).grid(row=1, column=2, padx=8, pady=10)

        dlg.wait_window()
        if result["mode"] == "file" and result["path"]:
            return result
        if result["mode"] == "folder" and result["folder"]:
            return result
        return None

