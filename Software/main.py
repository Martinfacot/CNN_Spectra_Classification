import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import numpy as np
from SpectraInterface import SpectraInterface
from symplified_code_to_run_model import predict_image
from OverviewPage import OverviewPage

class HomePage:
    def __init__(self, root):
        self.root = root
        self.root.title("Spectra Analysis")
        self.root.geometry("800x600")

        self.batch_folder = None
        self.image_data = {}
        self.classification_results = {}

        self.create_widgets()

    def create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Create and place widgets
        ttk.Button(main_frame, text="Select", command=self.select_folder).grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        ttk.Button(main_frame, text="Display", command=self.display_spectra).grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        ttk.Button(main_frame, text="Overview", command=self.show_overview).grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        # Batches frame
        batches_frame = ttk.LabelFrame(main_frame, text="Batches")
        batches_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

        self.batches_listbox = tk.Listbox(batches_frame)
        self.batches_listbox.pack(expand=True, fill="both")

        # Configure grid weights
        main_frame.grid_columnconfigure((0, 1, 2), weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.batch_folder = folder_path
            self.process_folder(folder_path)

    def process_folder(self, folder_path):
        # Clear existing batch list and image data
        self.batches_listbox.delete(0, tk.END)
        self.image_data = {}

        # Process images in the folder
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            try:
                prediction_result = predict_image(image_path)
                probability = prediction_result['probability']
                result = prediction_result['result']
                self.image_data[image_file] = {'probability': probability}
                self.batches_listbox.insert(tk.END, f"{image_file}: {result}")
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")

    def display_spectra(self):
        if self.batch_folder and self.image_data:
            spectra_window = tk.Toplevel(self.root)
            SpectraInterface(spectra_window, self.batch_folder, self.image_data, self.save_classification_results)
        else:
            messagebox.showwarning("No Data", "No batch folder or image data available")

    def save_classification_results(self, results):
        self.classification_results = results
        messagebox.showinfo("Results Saved", "Classification results have been saved.")

    def show_overview(self):
        if self.classification_results:
            overview_window = tk.Toplevel(self.root)
            OverviewPage(overview_window, self.classification_results)
        else:
            messagebox.showwarning("No Results", "No classification results available. Please classify images first.")

def main():
    root = tk.Tk()
    HomePage(root)
    root.mainloop()

if __name__ == "__main__":
    main()