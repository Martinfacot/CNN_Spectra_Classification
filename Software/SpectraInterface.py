import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
import os

class SpectraInterface:
    def __init__(self, root, batch_folder, image_data, save_callback):
        self.root = root
        self.root.title("Spectra Interface")
        
        # Set a minimum size for the main window
        self.root.minsize(1200, 600)  # Increased width to accommodate 4 images per row

        # Store the batch folder path and image data
        self.batch_folder = batch_folder
        self.image_data = image_data
        self.save_callback = save_callback

        # Store classification results
        self.classification_results = {}

        # Create custom styles
        self.create_styles()

        # Main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # Navigation buttons
        ttk.Button(main_frame, text="Previous", command=self.previous_batch).grid(row=0, column=0, pady=10)
        ttk.Label(main_frame, text=f"Current Batch:\n{os.path.basename(self.batch_folder)}").grid(row=0, column=1, pady=10)
        ttk.Button(main_frame, text="Next", command=self.next_batch).grid(row=0, column=2, pady=10)

        # Save Results button
        ttk.Button(main_frame, text="Save Results", command=self.save_results).grid(row=0, column=3, pady=10)

        # Create a canvas with a scrollbar
        self.canvas = tk.Canvas(main_frame)
        self.scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Grid layout for canvas and scrollbar
        self.canvas.grid(row=1, column=0, columnspan=4, sticky="nsew")
        self.scrollbar.grid(row=1, column=4, sticky="ns")

        # Make the canvas expandable
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_columnconfigure(2, weight=1)
        main_frame.grid_columnconfigure(3, weight=1)

        # Populate the scrollable frame with image frames
        self.populate_scrollable_frame()

        self.fullscreen_window = None

        # Bind mousewheel to scroll
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def create_styles(self):
        style = ttk.Style()
        style.configure("Custom.TCheckbutton", padding=5, font=('Helvetica', 10))
        style.configure("Valid.TCheckbutton", background="green")
        style.configure("Invalid.TCheckbutton", background="red")

    def populate_scrollable_frame(self):
        for i, (image_file, image_info) in enumerate(self.image_data.items()):
            row = i // 4
            col = i % 4
            frame = ttk.Frame(self.scrollable_frame, borderwidth=2, relief="solid", width=280, height=260)  # Reduced height
            frame.grid(row=row, column=col, padx=5, pady=5)
            frame.grid_propagate(False)
            self.display_image(frame, image_file, image_info)

    def display_image(self, parent, image_file, image_info):
        image_path = os.path.join(self.batch_folder, image_file)
        
        # Open the image
        img = Image.open(image_path)
        img = img.resize((260, 200), Image.Resampling.LANCZOS)

        # Add colored rectangle based on probability
        img_with_rect = self.add_probability_rectangle(img, image_info['probability'])

        # Convert image to ImageTk format
        img_tk = ImageTk.PhotoImage(img_with_rect)

        # Create a label to display the image
        img_label = ttk.Label(parent, image=img_tk)
        img_label.image = img_tk
        img_label.place(x=6, y=10)  # Fixed position

        # Bind click event to the image label
        img_label.bind("<Button-1>", lambda event, path=image_path: self.show_fullscreen(path, image_info['probability']))

        # Create and display checkbuttons below the image
        valid_var = tk.BooleanVar()
        invalid_var = tk.BooleanVar()
        self.create_custom_checkbutton(parent, "Valid", 60, 215, valid_var)
        self.create_custom_checkbutton(parent, "Invalid", 160, 215, invalid_var)

        # Store the variables in the classification_results dictionary
        self.classification_results[image_file] = {"valid": valid_var, "invalid": invalid_var}

    def add_probability_rectangle(self, img, probability):
        draw = ImageDraw.Draw(img)
        
        if probability >= 0.995:
            color = "green"
        elif 0.20 < probability < 0.995:
            color = "orange"
        else:
            color = "red"
        
        # Draw colored rectangle
        draw.rectangle([0, 0, 70, 23], fill=color)
        
        # Add probability text
        prob_text = f"{probability:.6f}"
        draw.text((10, 5), prob_text, fill="white")
        
        return img

    def create_custom_checkbutton(self, parent, text, x, y, var):
        cb = ttk.Checkbutton(parent, text=text, variable=var, style="Custom.TCheckbutton")
        cb.place(x=x, y=y)

        if text == "Valid":
            cb.config(command=lambda: self.update_checkbutton_color(cb, var, "Valid"))
        else:
            cb.config(command=lambda: self.update_checkbutton_color(cb, var, "Invalid"))

        return cb

    def update_checkbutton_color(self, checkbutton, var, button_type):
        if var.get():
            checkbutton.config(style=f"{button_type}.TCheckbutton")
        else:
            checkbutton.config(style="Custom.TCheckbutton")

    def show_fullscreen(self, image_path, probability):
        if self.fullscreen_window:
            self.fullscreen_window.destroy()

        self.fullscreen_window = tk.Toplevel(self.root)
        self.fullscreen_window.attributes('-fullscreen', True)

        # Open and resize the image to fit the screen
        img = Image.open(image_path)
        screen_width = self.fullscreen_window.winfo_screenwidth()
        screen_height = self.fullscreen_window.winfo_screenheight()
        img = img.resize((screen_width, screen_height), Image.Resampling.LANCZOS)

        # Add colored rectangle based on probability
        img_with_rect = self.add_probability_rectangle(img, probability)

        img_tk = ImageTk.PhotoImage(img_with_rect)

        label = ttk.Label(self.fullscreen_window, image=img_tk)
        label.image = img_tk
        label.pack(fill=tk.BOTH, expand=True)

        # Bind Escape key to close fullscreen
        self.fullscreen_window.bind("<Escape>", lambda event: self.fullscreen_window.destroy())

    def previous_batch(self):
        print("Previous batch")

    def next_batch(self):
        print("Next batch")

    def save_results(self):
        results = {}
        for image_file, vars in self.classification_results.items():
            if vars["valid"].get() and vars["invalid"].get():
                messagebox.showerror("Invalid Classification", f"Image {image_file} cannot be both valid and invalid.")
                return
            elif vars["valid"].get():
                results[image_file] = "Valid"
            elif vars["invalid"].get():
                results[image_file] = "Invalid"
            else:
                results[image_file] = "Unclassified"
        
        self.save_callback(results)
        self.root.destroy()

# This part is for demonstration purposes only. In the actual implementation,
# this class will be instantiated from the main application.
if __name__ == "__main__":
    root = tk.Tk()
    # Example usage code here
    root.mainloop()