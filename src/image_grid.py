import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

def create_image_grid(images):
    root = tk.Tk()
    root.title("Choose what you like.")

    root.geometry()

    # Create a main frame to hold the grid of images
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill="both")

    style = ttk.Style()
    style.configure("Selected.TButton", background="green")
    
    selection = [False] * len(images)  # Initialize selection array
    
    def toggle_select(index):
        selection[index] = not selection[index]
        if selection[index]:
            buttons[index].config(style="Selected.TButton")
        else:
            buttons[index].config(style="TButton")
    
    buttons = []
    for i, img in enumerate(images):
        img = img.resize((200, 200))  # Resize for display
        photo = ImageTk.PhotoImage(img)
        btn = ttk.Button(main_frame, image=photo, command=lambda i=i: toggle_select(i), style="TButton")
        btn.image = photo  # Keep a reference!
        btn.grid(row=i//4, column=i%4)
        buttons.append(btn)
    
    # Place the main frame in the center of the window
    main_frame.grid_rowconfigure(0, weight=1)
    for i in range(4):
        main_frame.grid_columnconfigure(i, weight=1)

    # Add a frame for the submit button to center it
    submit_frame = tk.Frame(root)
    submit_frame.pack(fill="x", pady=20)
    submit_btn = ttk.Button(submit_frame, text="Submit", command=root.destroy)
    submit_btn.pack(pady=20)
    
    root.mainloop()
    
    return selection

if __name__ == "__main__":
    # Assuming you have an array of PIL Image objects named `images`
    images = [Image.new("RGB", (512, 512), color=(i*15, i*10, i*5)) for i in range(16)]  # Example images
    selection_result = create_image_grid(images)
    print(selection_result)
    breakpoint()
