import numpy as np
import json
import time
import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading


def import_data(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def calculate_distances(data, params):
    """Pre-calculate distances for the entire dataset"""
    for entry in data:
        entry['calculated_distance'] = abs(entry['obs_hight']*params[0])/(entry['obs_hight']*params[1] -
                                                                          entry['measured_hight']*params[2])*(entry['obs_hight']*params[3]+entry['relative_hight']*params[4])


def save_model(params, filename="model_params.json"):
    """Save model parameters and weights to a file"""
    model_data = {
        "params": params.tolist(),
    }

    with open(filename, 'w') as f:
        json.dump(model_data, f, indent=4)
    print(f"Model saved to {filename}")


def load_model(filename="model_params.json"):
    """Load model parameters and weights from a file"""
    if not os.path.exists(filename):
        print(f"Model file {filename} not found. Using default parameters.")
        return None

    try:
        with open(filename, 'r') as f:
            model_data = json.load(f)

        params = np.array(model_data["params"])

        print(f"Model loaded from {filename}")
        print(f"Parameters: {params}")

        return params
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


class DistanceMeasurementGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Distance Measurement System")
        self.root.geometry("600x500")

        # Model variables
        self.params = None
        self.model_loaded = False

        self.create_widgets()
        self.check_existing_model()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(main_frame, text="Distance Measurement System",
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Status label
        self.status_label = ttk.Label(main_frame, text="No model loaded",
                                      foreground="red")
        self.status_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))

        # Buttons
        self.train_button = ttk.Button(main_frame, text="Train Model",
                                       command=self.train_model_gui, width=20)
        self.train_button.grid(row=2, column=0, pady=5, padx=5)

        self.use_button = ttk.Button(main_frame, text="Use Model for Measurement",
                                     command=self.show_measurement_interface, width=20)
        self.use_button.grid(row=2, column=1, pady=5, padx=5)

        # Training progress
        self.progress_frame = ttk.LabelFrame(
            main_frame, text="Training Progress", padding="10")
        self.progress_frame.grid(
            row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        self.progress_var = tk.StringVar(value="No training in progress")
        self.progress_label = ttk.Label(
            self.progress_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Log text area
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.grid(row=4, column=0, columnspan=2,
                       sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=10, width=60)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Exit button
        exit_button = ttk.Button(
            main_frame, text="Exit", command=self.root.quit, width=10)
        exit_button.grid(row=5, column=0, columnspan=2, pady=10)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

    def log_message(self, message):
        """Add a message to the log area"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def check_existing_model(self):
        """Check if an existing model is available"""
        if os.path.exists("model_params.json"):
            self.params = load_model()
            if self.params is not None:
                self.model_loaded = True
                self.status_label.config(
                    text="Model loaded successfully", foreground="green")
                self.use_button.config(state="normal")
                self.log_message("Existing model loaded successfully")
            else:
                self.status_label.config(
                    text="Error loading existing model", foreground="red")
                self.use_button.config(state="disabled")
        else:
            self.status_label.config(
                text="No model found - please train first", foreground="red")
            self.use_button.config(state="disabled")

    def train_model_gui(self):
        """Start model training in a separate thread"""
        self.train_button.config(state="disabled")
        self.progress_var.set("Training started...")
        self.log_message("Starting model training...")

        # Start training in a separate thread to prevent GUI freezing
        training_thread = threading.Thread(target=self.train_model_thread)
        training_thread.daemon = True
        training_thread.start()

    def train_model_thread(self):
        """Training thread function"""
        try:
            self.params = self.train_model_gui_version()
            self.model_loaded = True

            # Update GUI on main thread
            self.root.after(0, self.training_complete)

        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            self.root.after(0, lambda: self.training_failed(error_msg))

    def training_complete(self):
        """Called when training completes successfully"""
        self.progress_var.set("Training completed successfully!")
        self.status_label.config(
            text="Model trained and ready", foreground="green")
        self.train_button.config(state="normal")
        self.use_button.config(state="normal")
        self.log_message("Training completed successfully!")
        messagebox.showinfo("Training Complete",
                            "Model training completed successfully!")

    def training_failed(self, error_msg):
        """Called when training fails"""
        self.progress_var.set("Training failed")
        self.train_button.config(state="normal")
        self.log_message(error_msg)
        messagebox.showerror("Training Failed", error_msg)

    def train_model_gui_version(self):
        """Train the model with GUI logging"""
        # Configuration
        MAX_ITERATIONS = 100000
        PARAM_MUTATION = 0.01
        best_params = None

        # Import data
        self.root.after(0, lambda: self.log_message(
            "Loading training data..."))
        all_data = import_data('data.jsonl')

        # Load existing model if available
        existing_params = load_model()
        existing_error = float('inf')

        # Evaluate existing model if available
        if existing_params is not None:
            self.root.after(0, lambda: self.log_message(
                "Evaluating existing model..."))
            calculate_distances(all_data, existing_params)

            total_error = 0
            for entry in all_data:

                error = (abs(entry['true_distance'] -
                         entry['calculated_distance']))
                total_error += abs(error)

            existing_error = total_error / len(all_data)
            self.root.after(0, lambda: self.log_message(
                f"Existing model error: {existing_error:.2f}"))
            best_params = existing_params.copy()
        else:
            self.root.after(0, lambda: self.log_message(
                "No existing model found. Training new model from scratch."))
            best_params = np.array([0.43, 0.93, 1.0, 1.0, 1.0])

        start_time = time.time()

        while True:
            params = best_params.copy()
            best_avg_error = float('inf')

            for iteration in range(MAX_ITERATIONS):
                params += np.random.normal(0, PARAM_MUTATION, size=5)

                # Pre-calculate distances with new parameters
                calculate_distances(all_data, params)

                # Train on all data
                total_error = 0
                for entry in all_data:

                    error = (abs(entry['true_distance'] -
                             entry['calculated_distance']))
                    total_error += abs(error)

                avg_error = total_error / len(all_data)

                # Check if this is better than our best so far
                if avg_error < best_avg_error:
                    best_avg_error = avg_error
                    iteration -= 10000
                    best_params = params.copy()
                    self.root.after(0, lambda e=best_avg_error, i=iteration:
                                    self.log_message(f"Iteration {i}: New best error: {e:.2f}"))

                    # Compare to existing model
                    if best_avg_error < existing_error:
                        self.root.after(0, lambda:
                                        self.log_message(f"New model outperforms existing model! ({best_avg_error:.2f} vs {existing_error:.2f})"))
                # Update weights if error is above threshold
                params = best_params.copy()

            # Break the outer while True loop if we've achieved good enough performance
            # AND it's better than the existing model
            if best_avg_error < existing_error:
                self.root.after(0, lambda: self.log_message(
                    "Target error achieved and better than existing model!"))
                break

            self.root.after(0, lambda: self.log_message(
                "Restarting with new random initialization..."))

        # Final results
        elapsed_time = time.time() - start_time
        self.root.after(0, lambda: self.log_message(
            f"Training completed in {elapsed_time:.2f} seconds"))
        self.root.after(0, lambda: self.log_message(
            f"Best parameters: {best_params}"))
        self.root.after(0, lambda: self.log_message(
            f"Best error: {best_avg_error:.2f}"))
        self.root.after(0, lambda: self.log_message(
            f"Previous model error: {existing_error:.2f}"))
        self.root.after(0, lambda: self.log_message(
            f"Improvement: {existing_error - best_avg_error:.2f}"))

        # Only save if better than existing model
        if best_avg_error < existing_error:
            save_model(best_params)
            self.root.after(0, lambda: self.log_message(
                "New model saved as it outperforms the existing model."))
            return best_params
        else:
            self.root.after(0, lambda: self.log_message(
                "Training didn't improve on existing model. Keeping existing model."))
            return existing_params

    def show_measurement_interface(self):
        """Show the measurement input interface"""
        if not self.model_loaded:
            messagebox.showerror(
                "Error", "No model available. Please train the model first.")
            return

        # Create measurement window
        measure_window = tk.Toplevel(self.root)
        measure_window.title("Distance Measurement")
        measure_window.geometry("350x370")

        # Input fields
        main_frame = ttk.Frame(measure_window, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(main_frame, text="Distance Measurement Tool",
                  font=('Arial', 14, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Input fields
        ttk.Label(main_frame, text="Measured Height (m):").grid(
            row=1, column=0, sticky=tk.W, pady=5)
        measured_height_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=measured_height_var,
                  width=15).grid(row=1, column=1, pady=5)

        ttk.Label(main_frame, text="Observation Height (m):").grid(
            row=2, column=0, sticky=tk.W, pady=5)
        obs_height_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=obs_height_var,
                  width=15).grid(row=2, column=1, pady=5)

        ttk.Label(main_frame, text="Relative Height (m):").grid(
            row=3, column=0, sticky=tk.W, pady=5)
        rel_height_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=rel_height_var,
                  width=15).grid(row=3, column=1, pady=5)

        # Results
        result_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        result_frame.grid(row=4, column=0, columnspan=2,
                          sticky=(tk.W, tk.E), pady=20)

        formula_result_var = tk.StringVar(value="Formula result: -")
        ttk.Label(result_frame, textvariable=formula_result_var).grid(
            row=0, column=0, sticky=tk.W)

        def calculate_distance():
            try:
                measured_height = float(measured_height_var.get())
                height = float(obs_height_var.get())
                relative_height = float(rel_height_var.get())

                # Calculate using formula
                calculated_distance = abs(height*self.params[0])/(height*self.params[1]-measured_height*self.params[2])*(
                    height*self.params[3]+relative_height*self.params[4])

                # Update results
                formula_result_var.set(
                    f"Formula result: {calculated_distance:.2f} meters")

            except ValueError:
                messagebox.showerror(
                    "Error", "Please enter valid numbers for all fields.")
            except Exception as e:
                messagebox.showerror("Error", f"Calculation error: {str(e)}")

        def clear_fields():
            measured_height_var.set("")
            obs_height_var.set("")
            rel_height_var.set("")
            formula_result_var.set("Formula result: -")

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=20)

        ttk.Button(button_frame, text="Calculate",
                   command=calculate_distance).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Clear", command=clear_fields).grid(
            row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Close", command=measure_window.destroy).grid(
            row=0, column=2, padx=5)


if __name__ == "__main__":
    # Create and run the GUI
    root = tk.Tk()
    app = DistanceMeasurementGUI(root)
    root.mainloop()