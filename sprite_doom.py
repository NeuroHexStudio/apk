# sprite_doom.py
import os
import threading
import warnings
import numpy as np
from PIL import Image, ImageEnhance, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

warnings.filterwarnings("ignore", category=DeprecationWarning)

# -------------------------
# Theme (no emojis, professional)
# -------------------------
THEME = {
    "bg": "#120b09",
    "fg": "#ff4444",
    "accent1": "#8b0000",
    "accent2": "#ff6666",
    "button_bg": "#2f1b1b",
    "button_fg": "#ffffff",
    "button_hover": "#5a2828",
    "checker1": "#211010",
    "checker2": "#170505",
    "title": "BRUTAL DOOM — Palette Shader",
    "canvas_border": "#8b0000",
    "scale_trough": "#3d1f1f"
}

# -------------------------
# Built-in palette setup
# -------------------------
BRUTAL_DOOM_PALETTE = np.array([
    [24,0,0],[48,0,0],[72,8,0],[96,16,0],[128,32,0],[160,48,0],[192,64,0],[224,80,0],
    [255,96,0],[255,128,32],[192,96,32],[160,72,16],[96,48,16],[64,32,8],
    [224,160,128],[192,128,96],[160,96,64],[128,72,48],[96,48,32],
    [255,255,255],[224,224,224],[192,192,192],[160,160,160],[128,128,128],
    [96,96,96],[64,64,64],[32,32,32],[0,0,0],
    [0,32,0],[0,64,0],[0,96,0],[0,128,0],[0,160,0],[32,192,32],[64,224,64],[96,255,96],
    [0,0,64],[0,0,96],[0,0,128],[0,0,160],[0,0,192],[32,32,224],[64,64,255],[128,128,255],
    [128,96,0],[160,128,0],[192,160,0],[224,192,0],[255,224,0],[255,192,64],[255,160,32],
    [255,128,16],[255,96,0],
    [64,0,0],[96,0,0],[128,0,0],[160,0,0],[192,0,0],[224,0,0],[255,0,0],[255,32,32],[255,64,64],
    [32,32,64],[48,48,96],[72,72,128],[96,96,160],[128,128,192],[160,160,224],[192,192,255],
    [255,255,192],[255,224,160],[255,192,128],[224,160,96],[192,128,64],[160,96,32],
    [128,64,16],[96,48,8],[80,40,0],[48,24,0],[16,8,0]
], dtype=np.uint8)

SHADES_PER_COLOR = 6
SHADE_FACTORS = np.linspace(0.28, 1.0, SHADES_PER_COLOR)

def build_palette_shades(palette):
    p = palette.astype(np.float32) / 255.0
    shades = []
    for c in p:
        variants = []
        for f in SHADE_FACTORS:
            v = np.clip(c * f, 0.0, 1.0)
            variants.append((v * 255.0).astype(np.uint8))
        shades.append(variants)
    return np.array(shades, dtype=np.uint8)

PALETTE_SHADES = build_palette_shades(BRUTAL_DOOM_PALETTE)

# -------------------------
# Core shading engine (unchanged)
# -------------------------
class BrutalDoomShadingFilter:
    def __init__(self, palette=BRUTAL_DOOM_PALETTE, palette_shades=PALETTE_SHADES):
        self.palette = palette.astype(np.float32)
        self.palette_shades = palette_shades.astype(np.uint8)
        self.n_colors = self.palette.shape[0]
        self.n_shades = self.palette_shades.shape[1]

    @staticmethod
    def luminance_rgb(rgb):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        return (0.2126 * r + 0.7152 * g + 0.0722 * b)

    def find_nearest_palette_idx(self, pixels):
        dif = pixels[:, None, :] - self.palette[None, :, :]
        d2 = np.sum(dif * dif, axis=2)
        return np.argmin(d2, axis=1)

    def shade_index_from_luma(self, luma):
        mapped = np.clip(luma, 0.0, 1.0) ** 1.2
        idx = (mapped * (self.n_shades - 1)).astype(np.int32)
        return np.clip(idx, 0, self.n_shades - 1)

    def remap_image(self, img_pil, hybrid=True, intensity=0.8):
        has_alpha = img_pil.mode in ('RGBA', 'LA') or (img_pil.mode == 'P' and 'transparency' in img_pil.info)
        alpha_channel = np.array(img_pil.convert("RGBA"))[:, :, 3] if has_alpha else None

        arr = np.array(img_pil.convert("RGB"), dtype=np.uint8)
        h, w, _ = arr.shape
        flat = arr.reshape(-1, 3).astype(np.float32)

        pre_dark_gamma = 1.0 + (0.08 * intensity)
        flat_dark = ((flat / 255.0) ** pre_dark_gamma) * 255.0
        flat_dark = np.clip(flat_dark, 0, 255).astype(np.float32)

        nearest_idx = self.find_nearest_palette_idx(flat_dark)
        luma = self.luminance_rgb(flat / 255.0)
        shade_idx = self.shade_index_from_luma(luma)
        shades = self.palette_shades[nearest_idx, shade_idx]

        if hybrid:
            alpha = 0.35 + 0.6 * intensity
            blended = (1.0 - alpha) * flat + alpha * shades.astype(np.float32)
            out_flat = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            out_flat = shades.astype(np.uint8)

        out_arr = out_flat.reshape(h, w, 3)
        if alpha_channel is not None:
            out_img = Image.fromarray(out_arr).convert("RGBA")
            out_img.putalpha(Image.fromarray(alpha_channel))
        else:
            out_img = Image.fromarray(out_arr).convert("RGB")

        out_img = ImageEnhance.Sharpness(out_img).enhance(1.15 + 0.25 * intensity)
        out_img = ImageEnhance.Contrast(out_img).enhance(1.03 + 0.22 * intensity)
        return out_img

# -------------------------
# Small color helpers for animation
# -------------------------
def hex_to_rgb(hexcolor):
    h = hexcolor.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*[max(0, min(255, int(v))) for v in rgb])

def lerp_color(a, b, t):
    return tuple(a_i + (b_i - a_i) * t for a_i, b_i in zip(a, b))

# -------------------------
# Animated button (subtle fade on hover)
# -------------------------
class AnimatedButton(tk.Button):
    def __init__(self, master, bg, hover_bg, fg, **kwargs):
        super().__init__(master, bg=bg, fg=fg, activebackground=hover_bg, relief="flat", **kwargs)
        self._normal = bg
        self._hover = hover_bg
        self._anim_after = None
        self.bind("<Enter>", self._start_hover)
        self.bind("<Leave>", self._end_hover)

    def _start_hover(self, _):
        self._animate_to(self._hover)

    def _end_hover(self, _):
        self._animate_to(self._normal)

    def _animate_to(self, target_hex, steps=8, delay=16):
        if self._anim_after:
            try:
                self.after_cancel(self._anim_after)
            except Exception:
                pass
        start = hex_to_rgb(self["bg"])
        target = hex_to_rgb(target_hex)
        step = 0

        def stepper():
            nonlocal step
            t = (step + 1) / steps
            col = lerp_color(start, target, t)
            self.config(bg=rgb_to_hex(col))
            step += 1
            if step < steps:
                self._anim_after = self.after(delay, stepper)
            else:
                self._anim_after = None
                self.config(bg=target_hex)
        stepper()

# -------------------------
# GUI
# -------------------------
class SpriteDoomGUI:
    def __init__(self, root):
        self.root = root
        self.theme = THEME
        self.root.title(THEME["title"])
        self.root.geometry("900x700")
        self.root.configure(bg=self.theme["bg"])
        self.root.minsize(800, 600)

        try:
            self.root.iconbitmap("icon.ico")
        except Exception:
            pass

        self.filter = BrutalDoomShadingFilter()
        self.current_path = None
        self.result_image = None
        self.save_folder = None
        self.preview_photo = None

        # For block animations (store after ids so we can cancel)
        self._block_anim_after = {}

        self.build_ui()

    def build_ui(self):
        main = tk.Frame(self.root, bg=self.theme["bg"])
        main.pack(fill="both", expand=True, padx=10, pady=10)

        tk.Label(main, text=self.theme["title"], font=("Courier New", 16, "bold"),
                 fg=self.theme["fg"], bg=self.theme["bg"]).pack(pady=(0, 15))

        control = tk.Frame(main, bg=self.theme["bg"], relief="raised", bd=2)
        control.pack(fill="x", pady=(0, 15), padx=5)

        row = tk.Frame(control, bg=self.theme["bg"])
        row.pack(fill="x", pady=5)

        select_btn = AnimatedButton(row,
                                    text="Select Image",
                                    command=self.select_image,
                                    bg=self.theme["button_bg"],
                                    hover_bg=self.theme["button_hover"],
                                    fg=self.theme["button_fg"],
                                    font=("Courier New", 10, "bold"),
                                    padx=12, pady=6)
        select_btn.pack(side="left", padx=5)

        folder_btn = AnimatedButton(row,
                                    text="Save Folder",
                                    command=self.choose_folder,
                                    bg=self.theme["button_bg"],
                                    hover_bg=self.theme["button_hover"],
                                    fg=self.theme["button_fg"],
                                    font=("Courier New", 10, "bold"),
                                    padx=12, pady=6)
        folder_btn.pack(side="left", padx=5)

        tk.Label(control, text="Shading intensity:", font=("Courier New", 9, "bold"),
                 bg=self.theme["bg"], fg=self.theme["fg"]).pack(anchor="w", padx=10)
        self.int_var = tk.DoubleVar(value=0.75)
        tk.Scale(control, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                 variable=self.int_var, length=400, bg=self.theme["bg"], fg=self.theme["fg"],
                 troughcolor=self.theme["scale_trough"]).pack(fill="x", padx=10)

        apply_btn = AnimatedButton(control,
                                   text="Apply Shading",
                                   command=self.apply_hybrid,
                                   bg=self.theme["accent1"],
                                   hover_bg=self.theme["accent2"],
                                   fg=self.theme["button_fg"],
                                   font=("Courier New", 12, "bold"),
                                   padx=20, pady=8)
        apply_btn.pack(pady=10)

        self.status = tk.Label(control, text="Select an image and press Apply",
                              bg=self.theme["bg"], fg=self.theme["fg"], font=("Courier New", 9, "bold"))
        self.status.pack(pady=5)

        preview = tk.Frame(main, bg=self.theme["bg"])
        preview.pack(fill="both", expand=True, pady=10)

        tk.Label(preview, text="[ PREVIEW - CLICK TO SAVE ]",
                 bg=self.theme["bg"], fg=self.theme["fg"], font=("Courier New", 11, "bold")).pack(pady=(0, 5))

        canvas_frame = tk.Frame(preview, bg=self.theme["checker2"], bd=3, relief="sunken")
        canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(canvas_frame, bg=self.theme["checker2"], highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Bind resizing and fullscreen handlers
        self.canvas.bind("<Configure>", self._draw_checkered_bg)
        self.root.bind("<Configure>", self._on_resize)
        self.root.bind("<F11>", lambda e: self.root.attributes("-fullscreen",
                                not self.root.attributes("-fullscreen")))
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))

        self.canvas.bind("<Enter>", self._on_preview_enter)
        self.canvas.bind("<Leave>", self._on_preview_leave)
        # Canvas click -> save
        self.canvas.bind("<Button-1>", self._on_preview_click)

        # Overlay label (for "Click to Save") — forward clicks
        self.overlay = tk.Label(self.canvas, text="CLICK TO SAVE",
                                bg=self.theme["accent2"], fg=self.theme["button_fg"],
                                font=("Courier New", 10, "bold"), relief="raised", bd=2)
        self.overlay.place_forget()
        self.overlay.bind("<Button-1>", lambda e: self._on_preview_click(None))
        self.overlay.bind("<Enter>", lambda e: self._on_preview_enter(None))
        self.overlay.bind("<Leave>", lambda e: self._on_preview_leave(None))

    # -------------------------
    # Interactive checkerboard with smooth fades
    # -------------------------
    def _draw_checkered_bg(self, event=None):
        # Clear old checker but keep image if present (image uses tag "image")
        # delete only checker items
        self.canvas.delete("checker")
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        c1, c2 = self.theme["checker1"], self.theme["checker2"]
        size = 20
        for y in range(0, height, size):
            for x in range(0, width, size):
                color = c1 if (x // size + y // size) % 2 == 0 else c2
                tag = f"block_{x}_{y}"
                self.canvas.create_rectangle(x, y, x+size, y+size,
                                             fill=color, outline="", tags=("checker", tag))
                # Bind events; use default arg trick to capture tag
                self.canvas.tag_bind(tag, "<Enter>", lambda e, t=tag: self._start_block_fade(t, True))
                self.canvas.tag_bind(tag, "<Leave>", lambda e, t=tag: self._start_block_fade(t, False))
                self.canvas.tag_bind(tag, "<Button-1>", lambda e, t=tag: self._on_block_click(t))

    def _start_block_fade(self, tag, hover, steps=8, delay=18):
        """Animate a block from checker color -> accent2 (hover) or back (leave)."""
        # Cancel any existing animation for the tag
        if tag in self._block_anim_after:
            try:
                self.canvas.after_cancel(self._block_anim_after[tag])
            except Exception:
                pass

        coords = tag.split("_")
        if len(coords) != 3:
            return
        x, y = int(coords[1]), int(coords[2])
        base_color = self.theme["checker1"] if (x // 20 + y // 20) % 2 == 0 else self.theme["checker2"]
        start = hex_to_rgb(self.canvas.itemcget(tag, "fill") or base_color)
        target = hex_to_rgb(self.theme["accent2"] if hover else base_color)
        step_idx = 0

        def stepper():
            nonlocal step_idx
            t = step_idx / steps
            col = lerp_color(start, target, t)
            self.canvas.itemconfig(tag, fill=rgb_to_hex(col))
            step_idx += 1
            if step_idx <= steps:
                self._block_anim_after[tag] = self.canvas.after(delay, stepper)
            else:
                # final set and cleanup
                self.canvas.itemconfig(tag, fill=rgb_to_hex(target))
                self._block_anim_after.pop(tag, None)

        stepper()

    def _on_block_click(self, tag):
        # simple action: show coordinate in status
        parts = tag.split("_")
        if len(parts) == 3:
            x, y = int(parts[1]), int(parts[2])
            self.status.config(text=f"Clicked block at {x},{y}")

    # -------------------------
    # Core file actions
    # -------------------------
    def choose_folder(self):
        d = filedialog.askdirectory(title="Choose default save folder")
        if d:
            self.save_folder = d
            self.status.config(text=f"Save folder: {os.path.basename(d)}")

    def select_image(self):
        f = filedialog.askopenfilename(title="Select image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if f:
            self.current_path = f
            self.status.config(text=f"Loaded: {os.path.basename(f)}")
            self.preview_image(Image.open(f))

    # -------------------------
    # Threaded shading (non-blocking)
    # -------------------------
    def apply_hybrid(self):
        if not self.current_path:
            messagebox.showwarning("No image", "Select an image first.")
            return

        def worker():
            try:
                self.root.after(0, lambda: self.status.config(text="Processing... please wait..."))
                orig = Image.open(self.current_path)
                intensity = float(self.int_var.get())
                hybrid = self.filter.remap_image(orig, hybrid=True, intensity=intensity)
                self.result_image = hybrid
                self.root.after(0, lambda: [
                    self.preview_image(hybrid),
                    self.status.config(text="Shading complete. Click preview to save.")
                ])
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def preview_image(self, pil_img):
        """Safe preview: scale down only for rendering on canvas."""
        try:
            # keep image tag separate so checkerboard can be redrawn without losing it
            self.canvas.delete("image")
            w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if w < 50 or h < 50:
                w, h = 600, 400
            img = pil_img.copy()
            max_preview = 1024
            if img.width > max_preview or img.height > max_preview:
                scale = max_preview / max(img.width, img.height)
                img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
            # further fit inside canvas minus padding
            img.thumbnail((w - 40, h - 40), Image.Resampling.LANCZOS)
            img = img.convert("RGBA")
            photo = ImageTk.PhotoImage(img)
            # center the image
            self.canvas.create_image(w // 2, h // 2, image=photo, tags="image")
            # strong reference
            self.preview_photo = photo
            # show overlay
            if self.result_image:
                self.overlay.place(relx=0.5, rely=0.95, anchor="s")
        except Exception as e:
            self.status.config(text=f"Preview error: {e}")

    def _on_resize(self, event=None):
        """Keep rendered image when resizing/fullscreen."""
        if self.result_image:
            self.preview_image(self.result_image)
        else:
            # redraw checkerboard
            self._draw_checkered_bg()

    def _on_preview_enter(self, _):
        if self.result_image:
            # show overlay and highlight border
            self.overlay.place(relx=0.5, rely=0.95, anchor="s")
            self.canvas.config(highlightbackground=self.theme["accent2"])

    def _on_preview_leave(self, _):
        if not self.result_image:
            self.overlay.place_forget()
        self.canvas.config(highlightbackground=self.theme["canvas_border"])

    def _on_preview_click(self, event):
        """Save dialog invoked from either canvas click or overlay click."""
        if not self.result_image:
            return
        default = f"{os.path.splitext(os.path.basename(self.current_path))[0]}_brutal_doom.png"
        initialdir = self.save_folder or (os.path.dirname(self.current_path) if self.current_path else os.getcwd())
        save_path = filedialog.asksaveasfilename(
            title="Save Brutal Doom Image As...",
            defaultextension=".png",
            initialfile=default,
            initialdir=initialdir,
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if save_path:
            try:
                self.result_image.save(save_path, "PNG", optimize=True)
                self.status.config(text=f"Saved: {os.path.basename(save_path)}")
                messagebox.showinfo("Saved", f"Saved to:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Save Error", str(e))

def main():
    root = tk.Tk()
    app = SpriteDoomGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
