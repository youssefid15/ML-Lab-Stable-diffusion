Titre du projet
# Stable Diffusion Image Generator (CustomTkinter GUI)

This project provides a lightweight graphical interface to generate images from text prompts using a pretrained **Stable Diffusion** model. The GUI is built with `customtkinter`, and image generation is handled via Hugging Face's `diffusers` library.

---

## ✨ Features

- 🔤 Input a custom prompt via the GUI
- 🎨 Generate an image using Stable Diffusion
- 🖼️ Display the generated image
- 💾 Automatically save it as `generatedimage.png`

---

## 🧠 Source Code Overview

### `main.ipynb`

```python
from customtkinter import *  # Modern tkinter UI
from diffusers import StableDiffusionPipeline  # Hugging Face diffusion model
import torch  # PyTorch for model inference
from PIL import Image, ImageTk  # Image processing
import tkinter as tk  # Base UI

# Create the main application window
root = CTk()

# Input field for text prompt
prompt_entry = CTkEntry(root, width=400)
prompt_entry.pack()

# Function triggered on "Generate" button click
def generate():
    prompt = prompt_entry.get()  # Get the text from input
    image = pipe(prompt).images[0]  # Generate image
    image.save("generatedimage.png")  # Save image
    img = ImageTk.PhotoImage(image)  # Convert for display
    lmain.configure(image=img)  # Show on label
    lmain.image = img  # Keep reference to avoid GC

# Create the "Generate" button
generate_button = CTkButton(root, text="Generate", command=generate)
generate_button.pack()

# Label to display the image
lmain = tk.Label(root)
lmain.pack()

# Load the pretrained diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "C:/Users/Idris/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/2880f2ca379f41b0226444936bb7a6766a227587",  
    torch_dtype=torch.float32,
    use_safetensors=False
)

# Move model to CPU (remove `.float16` if not using GPU)
pipe.to("cpu")

# Start the GUI loop
root.mainloop()


🛠️ Development Environment Setup
1. thon Version
Ensure you have thon 3.8+ installed. You can check with:
thon --version

2. Create a Virtual Environment (Optional but Recommended)
thon -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Required Libraries:
pip install tkinter
pip install customtkinter
pip install torch diffusers customtkinter pillow

4. Model Setup
If you want to load the model from Hugging Face directly, replace "CompVis/stable-diffusion-v1-4" with the local path:
StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    use_auth_token="your_huggingface_token",
    torch_dtype=torch.float32,
    use_safetensors=False
)

🚀 Run the Application
thon main.ipynb
Type a prompt like: "a spaceship flying through a rainbow galaxy"
Click Generate
Your image will appear

📂 Project Structure
stable-diffusion-gui/
│
├── main.ipynb          # Main application script
├── README.md           # Project documentation
└── generatedimage.png  # Saved output (after generation)

📃 License
This project is released under the MIT License.

