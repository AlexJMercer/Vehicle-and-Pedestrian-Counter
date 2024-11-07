import tkinter as tk

import os
from tkinter import filedialog, ttk

class BuildGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Setup Zoning System")

        self.root.geometry("730x170")
        self.root.resizable(False, False)
        self.root.configure(bg="#2b2b2b")  # Dark background color


        # Initialize return values
        self.videoPath = None
        self.modelPath = None
        self.loadCoords = None
        self.loadLabels = None


        # Styling for dark mode
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TLabel", background="#2b2b2b", foreground="white", font=("Helvetica", 10))
        style.configure("TEntry", fieldbackground="#3d3d3d", foreground="white")
        style.configure("TCheckbutton", background="#2b2b2b", foreground="cyan", font=("Helvetica", 10))
        style.configure("TButton", background="#3d3d3d", foreground="cyan", font=("Helvetica", 10), borderwidth=1)

        # Button style with no hover effect
        style.map("TButton", background=[("active", "#ddd")])  # Disable hover color change
        style.map("TCheckbutton", background=[("active", "#2b2b2b")])  # Disable hover color change


        # Video Path label
        ttk.Label(root, text="Enter Video Path:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.videoPathEntry = ttk.Entry(root, width=50)
        self.videoPathEntry.grid(row=0, column=1, columnspan=3, padx=10, pady=5)
        
        # Browse Button for Video Path
        self.browseVideoButton = ttk.Button(root, text="Browse", command=lambda: self.browse_file("video"))
        self.browseVideoButton.grid(row=0, column=4, padx=10, pady=5)

        # Model Path label
        ttk.Label(root, text="Enter YOLO Path:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.modelPathEntry = ttk.Entry(root, width=50)
        self.modelPathEntry.grid(row=1, column=1, columnspan=3, padx=10, pady=5)
        
        # Browse Button for Model Path
        self.browseModelButton = ttk.Button(root, text="Browse", command=lambda: self.browse_file("model"))
        self.browseModelButton.grid(row=1, column=4, padx=10, pady=5)

        # Load existing coordinates
        self.loadCoordsVar = tk.BooleanVar()
        self.loadCoordsCheckbutton = ttk.Checkbutton(root, width=40, text="Load existing zone coordinates ?", variable=self.loadCoordsVar)
        self.loadCoordsCheckbutton.grid(row=2, column=0, sticky="w", padx=10, pady=5)

        # Load existing labels
        self.loadLabelsVar = tk.BooleanVar()
        self.loadLabelsCheckbutton = ttk.Checkbutton(root, width=40, text="Load existing zone labels ?", variable=self.loadLabelsVar)
        self.loadLabelsCheckbutton.grid(row=2, column=3, sticky="w", padx=10, pady=5)

        # Run Button
        self.runButton = ttk.Button(root, text="Run", command=self.run_app)
        self.runButton.grid(row=3, column=0, columnspan=5, pady=20)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(4, weight=1)

        # Bind the window close event to the on_closing method
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    
    # Handle window close event
    def on_closing(self):
        self.root.quit()
        self.root.destroy()
        exit()
    
    # Function to browse video file
    def browse_file(self, type):
        if type == "model":
            model_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("Model Files", "*.cfg;*.weights;*.pt")])
            if model_path:
                self.modelPathEntry.delete(0, tk.END)
                self.modelPathEntry.insert(0, model_path)
        
        elif type == "video":
            video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi")])
            if video_path:
                self.videoPathEntry.delete(0, tk.END)
                self.videoPathEntry.insert(0, video_path)
        
        else:
            print("Error: Invalid type")
            return

    # Main loop for the application 
    def run_app(self):
        # Retrieve and store the values in instance variables
        self.videoPath = self.videoPathEntry.get()
        self.modelPath = self.modelPathEntry.get()
        self.loadCoords = self.loadCoordsVar.get()
        self.loadLabels = self.loadLabelsVar.get()

        # Validate video path
        if self.videoPath == "0":
            self.videoPath = 0

        elif self.videoPath == "":
            print("Error: Video Path is empty")
            return
        
        elif not os.path.isfile(self.videoPath):
            print("Error: Video Path is invalid")
            return

        # Validate model path
        if self.modelPath == "":
            print("Error: Model Path is empty")
            return
        elif not os.path.isfile(self.modelPath):
            print("Error: Model Path is invalid")
            return

        # When run button is clicked, close GUI
        self.root.quit()
        self.root.destroy()

    # Return the values of the GUI
    def get_values(self):
        return self.videoPath, self.modelPath, self.loadCoords, self.loadLabels

