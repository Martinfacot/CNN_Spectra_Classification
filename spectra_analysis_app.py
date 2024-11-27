import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image, ImageTk
import json
import tensorflow as tf
from tensorflow.keras.metrics import Metric # type: ignore
from tensorflow.keras import backend as K # type: ignore
from tensorflow.keras.saving import register_keras_serializable # type: ignore

@register_keras_serializable()
class F1Score(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        self.true_positives.assign_add(tf.reduce_sum(y_pred * y_true))
        self.false_positives.assign_add(tf.reduce_sum(y_pred * (1 - y_true)))
        self.false_negatives.assign_add(tf.reduce_sum((1 - y_pred) * y_true))

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        return 2 * precision * recall / (precision + recall + K.epsilon())

    def reset_states(self):
        self.true_positives.assign(0.)
        self.false_positives.assign(0.)
        self.false_negatives.assign(0.)

class SpectraAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spectra Analysis Software")

        
        # Define color palette
        self.theme_colors = {
            'background': '#FFF5E1',   # Soft pastel orange
            'primary': '#FFD700',      # Pastel gold
            'secondary': '#FFDAB9',    # Peach
            'accent': '#FFA07A',       # Light Salmon
            'text': '#333333',         # Dark gray
            'button_bg': '#FFC46C',    # Soft orange
            'button_hover': '#FFB347', # Lighter orange
            'progress_bg': '#FFE4B5',  # Moccasin light
            'progress_fg': '#FF6347'   # Tomato red
        }
        
        # Configure theme
        self.configure_theme()
        
        # Store metabolite thresholds
        self.metabolite_thresholds = {
            'Valine': {'green': 0.99, 'orange': 0.10},
            'Isoleucine': {'green': 0.99, 'orange': 0.10},
            'Leucine': {'green': 0.99, 'orange': 0.10},
            'Alanine': {'green': 0.99, 'orange': 0.10}
        }
        
        # Store loaded data
        self.current_folder = None
        self.loaded_images = {}
        self.predictions = {}
        self.classifications = {}
        
        self.create_home_page()
        
    def configure_theme(self):
        style = ttk.Style()
        
        # Configure root window
        self.root.configure(bg=self.theme_colors['background'])
        
        # Configure default style
        style.configure('TFrame', background=self.theme_colors['background'])
        style.configure('TLabel', 
                        background=self.theme_colors['background'], 
                        foreground=self.theme_colors['text'],
                        font=('Helvetica', 10))
        
        # Button styles with rounded corners
        style.configure('TButton', 
                        background=self.theme_colors['button_bg'], 
                        foreground=self.theme_colors['text'],
                        font=('Helvetica', 10, 'bold'),
                        padding=5,
                        borderwidth=0,  # Remove default border
                        relief='flat')  # Flat style helps with custom look
        
        # Map style for different button states
        style.map('TButton', 
                background=[('active', self.theme_colors['button_hover']),
                            ('pressed', self.theme_colors['secondary'])],
                relief=[('pressed', 'sunken')])
        
        
        # LabelFrame style for patient folder frame
        style.configure('PatientFolder.TLabelframe', 
                        background=self.theme_colors['background'],
                        foreground=self.theme_colors['text'],
                        font=('Helvetica', 11, 'bold'))
        style.configure('PatientFolder.TLabelframe.Label', 
                        background=self.theme_colors['background'],
                        foreground=self.theme_colors['accent'])
        
        # Checkbutton styles
        style.configure('TCheckbutton', 
                        background=self.theme_colors['background'], 
                        foreground=self.theme_colors['text'])
        
        # Custom styles for metabolite buttons
        style.configure('Metabolite.TButton', 
                        background=self.theme_colors['secondary'], 
                        foreground=self.theme_colors['text'])
        style.configure('Selected.TButton', 
                        background=self.theme_colors['accent'], 
                        foreground=self.theme_colors['text'])
        style.configure('Green.TButton', 
                        background='#90EE90',  # Light green 
                        foreground=self.theme_colors['text'])
        
    def create_home_page(self):
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create buttons with improved spacing
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=0, column=0, columnspan=2, pady=20, sticky='ew')
        
        # Select patient folder button
        select_folder_btn = ttk.Button(button_frame, text="Select patient folder", command=self.select_folder)
        select_folder_btn.grid(row=0, column=0, padx=(0,20))
        
        # Display spectra button with spacing
        self.display_spectra_button = ttk.Button(button_frame, text="Display spectra with probability", 
                                                 command=self.show_spectra_interface, state='disabled')
        self.display_spectra_button.grid(row=0, column=1, padx=20)
        
        # Overview button with spacing
        self.overview_button = ttk.Button(button_frame, text="Overview", 
                                          command=self.show_overview, state='disabled')
        self.overview_button.grid(row=0, column=2, padx=(20,0))
        
        # Run all button
        self.run_all_button = ttk.Button(main_frame, text="Run all", 
                                         command=self.run_all_analysis, state='disabled')
        self.run_all_button.grid(row=3, column=0, columnspan=2, pady=20)
        
        # Create info frame with improved styling
        self.info_frame = ttk.LabelFrame(main_frame, text="Folder Information", 
                                         style='PatientFolder.TLabelframe')
        self.info_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # Create progress bar with custom style
        self.progress = ttk.Progressbar(main_frame, 
                                        style='Custom.Horizontal.TProgressbar', 
                                        orient=tk.HORIZONTAL, 
                                        length=400, 
                                        mode='determinate')
        self.progress.grid(row=5, column=0, columnspan=2, pady=20)
        
    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.current_folder = folder_path
            self.process_patient_folder(folder_path)
            
    def process_patient_folder(self, folder_path):
        try:
            deconv_folder = os.path.join(folder_path, "LED_Images", "Deconv")
            
            if not os.path.exists(deconv_folder):
                messagebox.showerror("Error", "Invalid folder structure")
                return
                
            # Clear existing info
            for widget in self.info_frame.winfo_children():
                widget.destroy()
                
            # Display folder ID
            folder_id = os.path.basename(folder_path)
            ttk.Label(self.info_frame, text=f"Folder ID: {folder_id}").grid(row=0, column=0, sticky=tk.W)
            
            # Load images for each metabolite
            self.loaded_images = {}
            row = 1
            for metabolite in self.metabolite_thresholds.keys():
                metabolite_folder = os.path.join(deconv_folder, metabolite)
                if os.path.exists(metabolite_folder):
                    images = os.listdir(metabolite_folder)
                    self.loaded_images[metabolite] = images
                    ttk.Label(self.info_frame, text=f"{metabolite}: {len(images)} images").grid(row=row, column=0, sticky=tk.W)
                    row += 1
                    
            # Enable buttons
            self.display_spectra_button.config(state='normal')
            self.overview_button.config(state='normal')
            self.run_all_button.config(state='normal')
            
        except Exception as e:
            messagebox.showerror("Error", f"Error processing folder: {str(e)}")
            
    def preprocess_image(self, image_path, coords=[70, 108], crop_size=[465, 605], resize_shape=[320, 320]):
        try:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Ensure the image has the expected dimensions
            if img.shape != (600, 800, 3):
                return None
                
            # Crop and resize the image
            cropped_img = img[coords[0]:(coords[0] + crop_size[0]), 
                            coords[1]:(coords[1] + crop_size[1])]
            resized_img = cv2.resize(cropped_img, resize_shape, interpolation=cv2.INTER_AREA)
            normalized_img = resized_img.astype(np.float32) / 255.0
            return normalized_img
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {str(e)}")
            return None
            
    def run_all_analysis(self):
        try:
            self.predictions = {}
            models_path = r"C:\Users\Biosfer\Desktop\CNN\software\models"
            
            total_images = sum(len(images) for images in self.loaded_images.values())
            processed_images = 0
            
            for metabolite in self.loaded_images:
                model_path = os.path.join(models_path, f"{metabolite.lower()}.keras")
                if not os.path.exists(model_path):
                    messagebox.showerror("Error", f"Model not found for {metabolite}")
                    continue
                    
                # Load model
                model = tf.keras.models.load_model(model_path, custom_objects={"F1Score": F1Score})

                """
                choice 1 : retrain all models with  using the custom f1 score from sklearn
                choice 2 : retrain all models with  using the custom f1 score from keras adding @register_keras_serializable to dont have to implement f1 score custom in this file
                The currently model with @register_keras_serializable are : acetate, acetone, glutamine, histdine.
                """

                metabolite_predictions = {}                                                         
                
                # Process each image
                for image_name in self.loaded_images[metabolite]:
                    image_path = os.path.join(self.current_folder, "LED_Images", "Deconv", 
                                            metabolite, image_name)
                    processed_image = self.preprocess_image(image_path)
                    
                    if processed_image is not None:
                        # Prepare the image for prediction
                        image_tensor = tf.convert_to_tensor(np.expand_dims(processed_image, axis=0), 
                                                         dtype=tf.float32)
                        prediction = model.predict(image_tensor)[0][0]
                        metabolite_predictions[image_name] = float(prediction)
                        
                    processed_images += 1
                    self.progress['value'] = (processed_images / total_images) * 100
                    self.root.update_idletasks()
                        
                self.predictions[metabolite] = metabolite_predictions
                
            messagebox.showinfo("Success", "Analysis completed successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during analysis: {str(e)}")

            
    def show_spectra_interface(self):
        # Create new window
        spectra_window = tk.Toplevel(self.root)
        spectra_window.title("Spectra Interface")
        spectra_window.geometry("1200x800")
        spectra_window.configure(bg=self.theme_colors['background'])
        
        # Create main frame with scrollbar
        main_frame = ttk.Frame(spectra_window, style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create metabolite buttons frame
        buttons_frame = ttk.Frame(main_frame, style='TFrame')
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create content frame with scrollbar
        canvas = tk.Canvas(main_frame, 
                           bg=self.theme_colors['background'], 
                           highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        content_frame = ttk.Frame(canvas, style='TFrame')
        
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        canvas.create_window((0, 0), window=content_frame, anchor=tk.NW)
        content_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

        # Dictionary to store metabolite buttons
        metabolite_buttons = {}
        
        # Function to reset button colors and display images
        def select_metabolite(metabolite):
            # Reset all button colors to default
            for met, btn in metabolite_buttons.items():
                # Check if this metabolite has been saved
                if hasattr(self, 'saved_classifications') and met in self.saved_classifications:
                    btn.configure(style='Green.TButton')
                else:
                    btn.configure(style='Metabolite.TButton')
            
            # Set selected button color to accent
            metabolite_buttons[metabolite].configure(style='Selected.TButton')
            
            # Display metabolite images
            self.display_metabolite_images(metabolite, content_frame)
        
        # Create metabolite buttons
        for metabolite in self.metabolite_thresholds.keys():
            btn = ttk.Button(buttons_frame, text=metabolite,
                             style='Metabolite.TButton',
                             command=lambda m=metabolite: select_metabolite(m))
            btn.pack(side=tk.LEFT, padx=2)
            metabolite_buttons[metabolite] = btn
        
        # Modify save_classifications method
        def save_classifications(metabolite):
            unclassified = []
            for image_name, vars in self.classifications[metabolite].items():
                if not vars['valid'].get() and not vars['invalid'].get():
                    unclassified.append(image_name)
            
            if unclassified:
                messagebox.showwarning("Warning", 
                                    f"Unclassified images found: {', '.join(unclassified)}")
                return
            
            # Save classifications
            results = {}
            for image_name, vars in self.classifications[metabolite].items():
                results[image_name] = 'valid' if vars['valid'].get() else 'invalid'
            
            # Store results 
            if not hasattr(self, 'saved_classifications'):
                self.saved_classifications = {}
            self.saved_classifications[metabolite] = results
            
            # Update button style to green border
            metabolite_buttons[metabolite].configure(style='Green.TButton')
            
            # Show success message without closing the window
            messagebox.showinfo("Success", f"Classifications saved for {metabolite}")
        
        # Bind the save_classifications method to the save button in display_metabolite_images
        def custom_save_classifications(m):
            save_classifications(m)
            # Optionally, you can force a refresh of the button styles
            select_metabolite(m)
        
        # Override the save method in display_metabolite_images
        self.save_classifications = custom_save_classifications
        
        # Display first metabolite by default and set its button to blue
        first_metabolite = list(self.metabolite_thresholds.keys())[0]
        select_metabolite(first_metabolite)
        
        # Bind mouse scroll to canvas
        def _on_mouse_wheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mouse_wheel)

    def show_full_screen_image(self, image_path):
        # Create new window for full-screen image
        full_screen_window = tk.Toplevel(self.root)
        full_screen_window.title("Full Screen Image")

        # Make window full screen
        full_screen_window.attributes('-fullscreen', True)

        # Add "Exit Full Screen" button
        ttk.Button(
            full_screen_window,
            text="Exit Full Screen",
            command=full_screen_window.destroy
        ).pack(pady=10)

        # Load and display image at full size
        img = Image.open(image_path)

        # Calculate scaling to fit screen while maintaining aspect ratio
        screen_width = full_screen_window.winfo_screenwidth()
        screen_height = full_screen_window.winfo_screenheight()
        img_ratio = img.size[0] / img.size[1]
        screen_ratio = screen_width / screen_height

        if screen_ratio > img_ratio:
            height = screen_height
            width = int(height * img_ratio)
        else:
            width = screen_width
            height = int(width / img_ratio)

        img = img.resize((width, height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)

        # Create label to display image
        img_label = ttk.Label(full_screen_window, image=photo)
        img_label.image = photo
        img_label.pack(expand=True)

        
    def get_probability_color(self, probability, metabolite):
        thresholds = self.metabolite_thresholds[metabolite]
        if probability >= thresholds['green']:
            return 'green'
        elif probability >= thresholds['orange']:
            return 'orange'
        else:
            return 'red'
            
    def display_metabolite_images(self, metabolite, frame):
        # Clear existing content
        for widget in frame.winfo_children():
            widget.destroy()
            
        if metabolite not in self.loaded_images or metabolite not in self.predictions:
            ttk.Label(frame, text=f"No data available for {metabolite}").pack()
            return
            
        # Create grid of images
        row = 0
        col = 0
        max_cols = 4
        
        for image_name in self.loaded_images[metabolite]:
            image_frame = ttk.Frame(frame, padding="5")
            image_frame.grid(row=row, column=col, padx=5, pady=5)
            
            # Load and display image
            image_path = os.path.join(self.current_folder, "LED_Images", "Deconv", 
                                    metabolite, image_name)
            img = Image.open(image_path)
            img = img.resize((320, 320))  # Thumbnail size
            photo = ImageTk.PhotoImage(img)
            
            # Create image label with probability indicator
            image_container = ttk.Frame(image_frame)
            image_container.pack()
            
            probability = self.predictions[metabolite][image_name]
            color = self.get_probability_color(probability, metabolite)
            
            prob_label = ttk.Label(image_container, 
                                 text=f"{probability:.3f}",
                                 background=color)
            prob_label.pack(anchor=tk.NW)
            
            # Make image clickable
            img_label = ttk.Label(image_container, image=photo, cursor="hand2")
            img_label.image = photo
            img_label.pack()
            
            # Bind click event to show full screen
            img_label.bind('<Button-1>', 
                         lambda e, path=image_path: self.show_full_screen_image(path))
            
            # Add checkboxes
            valid_var = tk.BooleanVar()
            invalid_var = tk.BooleanVar()
            
            # Set default checkbox states based on probability
            thresholds = self.metabolite_thresholds[metabolite]
            if probability >= thresholds['green']:
                valid_var.set(True)
            elif probability < thresholds['orange']:
                invalid_var.set(True)
                
            ttk.Checkbutton(image_frame, text="Valid", variable=valid_var).pack()
            ttk.Checkbutton(image_frame, text="Invalid", variable=invalid_var).pack()
            
            # Store classification variables
            if metabolite not in self.classifications:
                self.classifications[metabolite] = {}
            self.classifications[metabolite][image_name] = {
                'valid': valid_var,
                'invalid': invalid_var
            }
            
            # Update grid position
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
                
        # Add save button
        ttk.Button(frame, text=f"Save {metabolite} Classifications",
                  command=lambda m=metabolite: self.save_classifications(m)).grid(
                      row=row+1, column=0, columnspan=max_cols, pady=10)
                            
    def save_classifications(self, metabolite):
        unclassified = []
        invalid_classifications = []
        for image_name, vars in self.classifications[metabolite].items():
            if not vars['valid'].get() and not vars['invalid'].get():
                unclassified.append(image_name)
            elif vars['valid'].get() and vars['invalid'].get():
                invalid_classifications.append(image_name)
        
        if unclassified:
            parent_window = self.root.winfo_children()[-1]
            messagebox.showwarning("Warning", 
                                f"Unclassified images found: {', '.join(unclassified)}",
                                parent=parent_window)
            return
        
        if invalid_classifications:
            parent_window = self.root.winfo_children()[-1]
            messagebox.showwarning("Warning", 
                                f"Invalid classifications found (both valid and invalid checked): {', '.join(invalid_classifications)}",
                                parent=parent_window)
            return
        
        results = {}
        for image_name, vars in self.classifications[metabolite].items():
            results[image_name] = 'valid' if vars['valid'].get() else 'invalid'
        
        if not hasattr(self, 'saved_classifications'):
            self.saved_classifications = {}
        self.saved_classifications[metabolite] = results
        parent_window = self.root.winfo_children()[-1]
        messagebox.showinfo("Info", f"Classifications for {metabolite} saved successfully.", parent=parent_window)
        
    def show_overview(self):
        def filter_invalid_classifications():
            for item in tree.get_children():
                tree.delete(item)
            for metabolite, classifications in self.saved_classifications.items():
                for image_name, classification in classifications.items():
                    if classification == 'invalid':
                        tree.insert('', tk.END, values=(metabolite, image_name, classification))

        def show_all_classifications():
            for item in tree.get_children():
                tree.delete(item)
            for metabolite, classifications in self.saved_classifications.items():
                for image_name, classification in classifications.items():
                    tree.insert('', tk.END, values=(metabolite, image_name, classification))

        def sort_by_selected_metabolite():
            selected_metabolite = metabolite_combobox.get()
            for item in tree.get_children():
                tree.delete(item)
            for metabolite, classifications in self.saved_classifications.items():
                if metabolite == selected_metabolite:
                    for image_name, classification in classifications.items():
                        tree.insert('', tk.END, values=(metabolite, image_name, classification))

        # Create new window
        overview_window = tk.Toplevel(self.root)
        overview_window.title("Classification Overview")
        overview_window.geometry("800x600")

        # Create main frame with scrollbar
        main_frame = ttk.Frame(overview_window)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create treeview for results
        tree = ttk.Treeview(main_frame, columns=('Metabolite', 'Image', 'Classification'),
                            show='headings')
        tree.heading('Metabolite', text='Metabolite')
        tree.heading('Image', text='Image')
        tree.heading('Classification', text='Classification')

        # Add scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        # Pack widgets
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Populate treeview
        for metabolite, classifications in self.saved_classifications.items():
            for image_name, classification in classifications.items():
                tree.insert('', tk.END, values=(metabolite, image_name, classification))

        # Add filter and sort buttons
        filter_frame = ttk.Frame(overview_window)
        filter_frame.pack(pady=10)
        ttk.Button(filter_frame, text="Show Invalid Classifications", command=filter_invalid_classifications).pack(side=tk.LEFT, padx=5)
        ttk.Button(filter_frame, text="Show All Classifications", command=show_all_classifications).pack(side=tk.LEFT, padx=5)

        # Add combobox for sorting by metabolite
        metabolite_combobox = ttk.Combobox(filter_frame, values=list(self.saved_classifications.keys()))
        metabolite_combobox.pack(side=tk.LEFT, padx=5)
        ttk.Button(filter_frame, text="Sort by Metabolite", command=sort_by_selected_metabolite).pack(side=tk.LEFT, padx=5)

        # Add export button
        ttk.Button(overview_window, text="Export Results", command=self.export_results).pack(pady=10)
                    
    def export_results(self):
        if not hasattr(self, 'saved_classifications'):
            messagebox.showwarning("Warning", "No classifications to export")
            return
            
        try:
# Create DataFrame from results
            rows = []
            for metabolite, classifications in self.saved_classifications.items():
                for image_name, classification in classifications.items():
                    rows.append({
                        'Metabolite': metabolite,
                        'Image': image_name,
                        'Classification': classification
                    })
            
            df = pd.DataFrame(rows)
            
            # Ask user for save location
            file_path = filedialog.asksaveasfilename(
                defaultextension='.xlsx',
                filetypes=[("Excel files", "*.xlsx")],
                initialfile='classification_results.xlsx'
            )
            
            if file_path:
                df.to_excel(file_path, index=False)
                messagebox.showinfo("Success", "Results exported successfully")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting results: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SpectraAnalysisApp(root)
    root.mainloop()