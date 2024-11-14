import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image, ImageTk
import json

class SpectraAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spectra Analysis Software")
        self.root.geometry("800x600")
        
        # Store metabolite thresholds
        self.metabolite_thresholds = {
            'Acetone': {'green': 0.998, 'orange': 0.25},
            'Glutamine': {'green': 0.99, 'orange': 0.10},
            'Glycine': {'green': 0.99, 'orange': 0.10},
            'Histidine': {'green': 0.99, 'orange': 0.10},
            'Isoleucine': {'green': 0.99, 'orange': 0.10},
            'Leucine': {'green': 0.99, 'orange': 0.10}
        }
        
        # Store loaded data
        self.current_folder = None
        self.loaded_images = {}
        self.predictions = {}
        self.classifications = {}
        
        self.create_home_page()
        
    def create_home_page(self):
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create buttons
        ttk.Button(main_frame, text="Select patient folder", command=self.select_folder).grid(row=0, column=0, pady=5)
        ttk.Button(main_frame, text="Display spectra with probability", command=self.show_spectra_interface, state='disabled').grid(row=1, column=0, pady=5)
        ttk.Button(main_frame, text="Overview", command=self.show_overview, state='disabled').grid(row=2, column=0, pady=5)
        ttk.Button(main_frame, text="Run all", command=self.run_all_analysis, state='disabled').grid(row=3, column=0, pady=5)
        
        # Create info frame
        self.info_frame = ttk.LabelFrame(main_frame, text="Folder Information", padding="10")
        self.info_frame.grid(row=4, column=0, pady=10, sticky=(tk.W, tk.E))
        
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
                    images = [f for f in os.listdir(metabolite_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
                    self.loaded_images[metabolite] = images
                    ttk.Label(self.info_frame, text=f"{metabolite} images: {len(images)}").grid(row=row, column=0, sticky=tk.W)
                    row += 1
                    
            # Enable buttons
            for widget in self.root.winfo_children()[0].winfo_children():
                if isinstance(widget, ttk.Button):
                    widget.configure(state='normal')
                    
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
            
            for metabolite in self.loaded_images:
                model_path = os.path.join(models_path, f"{metabolite.lower()}.keras")
                if not os.path.exists(model_path):
                    messagebox.showerror("Error", f"Model not found for {metabolite}")
                    continue
                    
                # Load model
                model = tf.keras.models.load_model(model_path)
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
                        
                self.predictions[metabolite] = metabolite_predictions
                
            messagebox.showinfo("Success", "Analysis completed successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during analysis: {str(e)}")
            
    def show_spectra_interface(self):
        # Create new window
        spectra_window = tk.Toplevel(self.root)
        spectra_window.title("Spectra Interface")
        spectra_window.geometry("1200x800")
        
        # Create main frame with scrollbar
        main_frame = ttk.Frame(spectra_window)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create metabolite buttons
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        for metabolite in self.metabolite_thresholds.keys():
            ttk.Button(buttons_frame, text=metabolite,
                      command=lambda m=metabolite: self.display_metabolite_images(m, content_frame)
                      ).pack(side=tk.LEFT, padx=2)
            
        # Create content frame with scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        content_frame = ttk.Frame(canvas)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        canvas.create_window((0, 0), window=content_frame, anchor=tk.NW)
        content_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        
        # Display first metabolite by default
        self.display_metabolite_images(list(self.metabolite_thresholds.keys())[0], content_frame)

    def show_full_screen_image(self, image_path):
        # Create new window for full-screen image
        full_screen_window = tk.Toplevel(self.root)
        full_screen_window.title("Full Screen Image")
        
        # Make window full screen
        full_screen_window.attributes('-fullscreen', True)
        
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
        img_label.image = photo  # Keep a reference
        img_label.pack(expand=True)
        
        # Add return button
        return_btn = ttk.Button(full_screen_window, 
                              text="Return (Esc)", 
                              command=full_screen_window.destroy)
        return_btn.pack(pady=10)
        
        # Bind Escape key to close window
        def close_fullscreen(event=None):
            full_screen_window.destroy()
        
        full_screen_window.bind('<Escape>', close_fullscreen)
        
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
        max_cols = 8
        
        for image_name in self.loaded_images[metabolite]:
            image_frame = ttk.Frame(frame, padding="5")
            image_frame.grid(row=row, column=col, padx=5, pady=5)
            
            # Load and display image
            image_path = os.path.join(self.current_folder, "LED_Images", "Deconv", 
                                    metabolite, image_name)
            img = Image.open(image_path)
            img = img.resize((100, 100))  # Thumbnail size
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
            
        # Store results (could be saved to file here)
        if not hasattr(self, 'saved_classifications'):
            self.saved_classifications = {}
        self.saved_classifications[metabolite] = results
        
        # Show success message without closing the window
        messagebox.showinfo("Success", f"Classifications saved for {metabolite}")
        
    def show_overview(self):
            
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
                
        # Add export button
        ttk.Button(overview_window, text="Export Results",
                  command=self.export_results).pack(pady=10)
                  
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
