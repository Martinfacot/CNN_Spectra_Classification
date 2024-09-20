import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd

class OverviewPage:
    def __init__(self, root, classification_results):
        self.root = root
        self.root.title("Overview")
        self.root.geometry("800x600")

        self.classification_results = classification_results

        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Create table
        self.tree = ttk.Treeview(main_frame, columns=('Image', 'Classification'), show='headings')
        self.tree.heading('Image', text='Image')
        self.tree.heading('Classification', text='Classification')
        self.tree.column('Image', width=400)
        self.tree.column('Classification', width=100)
        self.tree.grid(row=0, column=0, columnspan=2, sticky='nsew')

        # Add a scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.grid(row=0, column=2, sticky='ns')
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Populate the table
        for image, classification in self.classification_results.items():
            self.tree.insert('', 'end', values=(image, classification))

        # Export button
        export_button = ttk.Button(main_frame, text="Export Results", command=self.export_results)
        export_button.grid(row=1, column=0, columnspan=2, pady=10)

        # Configure grid weights
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

    def export_results(self):
        # Ask user for file location
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        
        if file_path:
            try:
                # Convert results to DataFrame
                df = pd.DataFrame(list(self.classification_results.items()), columns=['Image', 'Classification'])
                
                # Export to Excel
                df.to_excel(file_path, index=False)
                
                messagebox.showinfo("Export Successful", f"Results exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Export Failed", f"An error occurred while exporting: {str(e)}")

# This part is for demonstration purposes only. In the actual implementation,
# this class will be instantiated from the main application.
if __name__ == "__main__":
    root = tk.Tk()
    # Example classification results
    classification_results = {
        "image1.png": "Valid",
        "image2.png": "Invalid",
        "image3.png": "Unclassified"
    }
    app = OverviewPage(root, classification_results)
    root.mainloop()