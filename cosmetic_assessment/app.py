import tkinter as tk

from ui.main_window import MainWindow


def main() -> None:
    root = tk.Tk()
    app = MainWindow(root)
    app.run()


if __name__ == "__main__":
    main()