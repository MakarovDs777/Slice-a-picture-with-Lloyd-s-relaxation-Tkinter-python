import os
import random
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2

DEFAULT_POINTS = 200
DEFAULT_LLOYD_ITERS = 3

# --- вспомогательные функции (как раньше) ---
def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не получилось загрузить изображение:\n{e}")
        return None

def subdiv_voronoi_facets(points, rect):
    x0, y0, x1, y1 = rect
    subdiv = cv2.Subdiv2D((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))
    try:
        facets, centers = subdiv.getVoronoiFacetList([])
    except Exception:
        try:
            ret, facets, centers = cv2.Subdiv2D.getVoronoiFacetList(subdiv, [])
        except Exception:
            return []
    poly_list = []
    for f in facets:
        if f is None or len(f) == 0:
            continue
        pts = np.array(f, dtype=np.float64)
        poly_list.append(pts)
    return poly_list

def polygon_centroid(pts):
    if len(pts) < 3:
        return pts.mean(axis=0)
    x = pts[:,0]; y = pts[:,1]
    a = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    A = a * 0.5
    if abs(A) < 1e-9:
        return pts.mean(axis=0)
    cx = np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6.0 * A)
    cy = np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6.0 * A)
    return np.array([cx, cy])

def clip_polygon_to_rect(pts, rect):
    def clip_edge(poly, edge):
        a,b,c = edge
        out = []
        if len(poly) == 0:
            return out
        prev = poly[-1]
        prev_inside = (a*prev[0] + b*prev[1] <= c + 1e-8)
        for cur in poly:
            cur_inside = (a*cur[0] + b*cur[1] <= c + 1e-8)
            if cur_inside:
                if not prev_inside:
                    dx = cur[0]-prev[0]; dy = cur[1]-prev[1]; denom = a*dx + b*dy
                    if abs(denom) > 1e-9:
                        t = (c - a*prev[0] - b*prev[1]) / denom
                        t = max(0.0, min(1.0, t))
                        inter = (prev[0] + dx*t, prev[1] + dy*t)
                        out.append(inter)
                out.append(tuple(cur))
            elif prev_inside:
                dx = cur[0]-prev[0]; dy = cur[1]-prev[1]; denom = a*dx + b*dy
                if abs(denom) > 1e-9:
                    t = (c - a*prev[0] - b*prev[1]) / denom
                    t = max(0.0, min(1.0, t))
                    inter = (prev[0] + dx*t, prev[1] + dy*t)
                    out.append(inter)
            prev = cur
            prev_inside = cur_inside
        return out

    x0,y0,x1,y1 = rect
    edges = [
        ( 1, 0, x1),  # x <= x1
        (-1, 0, -x0), # x >= x0
        (0, 1, y1),   # y <= y1
        (0,-1, -y0)   # y >= y0
    ]
    poly = [tuple(p) for p in pts]
    for e in edges:
        poly = clip_edge(poly, e)
        if not poly:
            break
    return np.array(poly, dtype=np.float64)

def lloyd_relaxation(points, rect, iters):
    x0,y0,x1,y1 = rect
    pts = points.copy()
    for _ in range(iters):
        facets = subdiv_voronoi_facets(pts, rect)
        new_pts = []
        for poly in facets:
            clipped = clip_polygon_to_rect(poly, rect)
            if len(clipped) == 0:
                continue
            c = polygon_centroid(clipped)
            c[0] = min(max(c[0], x0), x1-1e-6)
            c[1] = min(max(c[1], y0), y1-1e-6)
            new_pts.append(c)
        if len(new_pts) == 0:
            break
        pts = np.array(new_pts, dtype=np.float64)
    return pts

# --- НОВЫЙ: вырезка патчей и их случайное перемешивание без blur ---
def extract_patches(image, facets, rect):
    h, w, _ = image.shape
    patches = []
    for poly in facets:
        clipped = clip_polygon_to_rect(poly, rect)
        if clipped is None or len(clipped) < 3:
            continue
        pts = np.array(clipped, dtype=np.int32)
        mask_full = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_full, [pts], 255)
        ys, xs = np.where(mask_full == 255)
        if len(ys) == 0:
            continue
        y0, y1 = ys.min(), ys.max()+1
        x0, x1 = xs.min(), xs.max()+1
        patch = image[y0:y1, x0:x1].copy()
        mask_bbox = mask_full[y0:y1, x0:x1].copy()
        centroid = polygon_centroid(clipped)
        patches.append({
            "patch": patch,
            "mask_bbox": mask_bbox,
            "bbox": (x0, y0, x1, y1),
            "centroid": centroid,
            "poly_pts": clipped
        })
    return patches

def build_shuffled_mosaic(image, facets, rect):
    h, w, _ = image.shape
    patches = extract_patches(image, facets, rect)
    if len(patches) == 0:
        return image.copy()
    # Создаём список целевых ячеек (facets clipped) для полного покрытия
    targets = []
    for poly in facets:
        clipped = clip_polygon_to_rect(poly, rect)
        if clipped is None or len(clipped) < 3:
            continue
        pts = np.array(clipped, dtype=np.int32)
        mask_full = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_full, [pts], 255)
        ys, xs = np.where(mask_full == 255)
        if len(ys) == 0:
            continue
        y0, y1 = ys.min(), ys.max()+1
        x0, x1 = xs.min(), xs.max()+1
        target_mask_bbox = mask_full[y0:y1, x0:x1]
        targets.append({
            "bbox": (x0, y0, x1, y1),
            "mask_bbox": target_mask_bbox,
            "poly_pts": clipped,
            "centroid": polygon_centroid(clipped)
        })
    # Если численности различаются — подрезаем до min
    n = min(len(patches), len(targets))
    patches = patches[:n]
    targets = targets[:n]

    perm = np.random.permutation(n)
    out = np.zeros_like(image)

    # Для каждой целевой ячейки берем случайный источник и ресайзим его bbox->target bbox
    for i, tgt in enumerate(targets):
        src = patches[perm[i]]
        tx0, ty0, tx1, ty1 = tgt["bbox"]
        tw = tx1 - tx0
        th = ty1 - ty0
        if tw <= 0 or th <= 0:
            continue
        src_patch = src["patch"]
        # ресайз исходного патча в размеры целевой bbox, nearest -> без размытия
        resized = cv2.resize(src_patch, (tw, th), interpolation=cv2.INTER_NEAREST)
        # для маски используем маску целевой ячейки (чтобы заполнить именно форму целевой ячейки)
        tgt_mask = tgt["mask_bbox"]
        # если размеры маски не совпадают (редко) — ресайзим маску nearest
        if tgt_mask.shape != (th, tw):
            tgt_mask = cv2.resize(tgt_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
            _, tgt_mask = cv2.threshold(tgt_mask, 127, 255, cv2.THRESH_BINARY)
        mask_bool = (tgt_mask == 255)
        # вставляем
        dst_region = out[ty0:ty1, tx0:tx1]
        # вырезаем по маске
        dst_region[mask_bool] = resized[mask_bool]
        out[ty0:ty1, tx0:tx1] = dst_region

    # На случай оставшихся пикселей (если что-то не покрыто), заполним из оригинала там, где out==0
    zero_mask = np.all(out == 0, axis=2)
    out[zero_mask] = image[zero_mask]
    return out

# ---------- GUI приложение ----------
class LloydShuffleApp:
    def __init__(self, master):
        self.master = master
        master.title("Остроугольная мозаика — Shuffle Fragments")
        self.img = None
        self.display_imgtk = None
        self.result = None

        control = tk.Frame(master, padx=8, pady=8)
        control.pack(side="left", fill="y")

        tk.Button(control, text="Открыть изображение", command=self.open_image).pack(fill="x", pady=4)
        tk.Label(control, text="Число точек:").pack(anchor="w")
        self.entry_points = tk.Entry(control)
        self.entry_points.insert(0, str(DEFAULT_POINTS))
        self.entry_points.pack(fill="x", pady=2)

        tk.Label(control, text="Итераций Ллойда:").pack(anchor="w")
        self.entry_iters = tk.Entry(control)
        self.entry_iters.insert(0, str(DEFAULT_LLOYD_ITERS))
        self.entry_iters.pack(fill="x", pady=2)

        tk.Button(control, text="Сгенерировать (sharp, shuffled)", command=self.generate).pack(fill="x", pady=8)
        tk.Button(control, text="Сохранить на рабочий стол", command=self.save_to_desktop).pack(fill="x", pady=4)

        self.canvas_frame = tk.Frame(master, bd=2, relief="sunken")
        self.canvas_frame.pack(side="right", fill="both", expand=True)
        self.canvas = tk.Canvas(self.canvas_frame, bg="grey")
        self.canvas.pack(fill="both", expand=True)
        self.canvas_image_id = None

    def open_image(self):
        path = filedialog.askopenfilename(title="Выберите изображение",
                                          filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if not path:
            return
        arr = load_image(path)
        if arr is None:
            return
        self.img = arr
        self.show_image(self.img)

    def show_image(self, arr):
        h, w, _ = arr.shape
        pil = Image.fromarray(arr)
        max_display = 900
        scale = min(1.0, max_display / max(w, h))
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            pil = pil.resize(new_size, Image.LANCZOS)
        self.display_imgtk = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.config(width=self.display_imgtk.width(), height=self.display_imgtk.height())
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.display_imgtk)
        self.master.update_idletasks()

    def generate(self):
        if self.img is None:
            messagebox.showwarning("Внимание", "Откройте изображение сначала.")
            return
        try:
            n = int(self.entry_points.get())
            iters = int(self.entry_iters.get())
        except Exception:
            messagebox.showerror("Ошибка", "Параметры должны быть целыми числами.")
            return
        if n <= 0 or iters < 0:
            messagebox.showerror("Ошибка", "Неверные параметры.")
            return

        h, w, _ = self.img.shape
        pts = np.zeros((n,2), dtype=np.float64)
        for i in range(n):
            pts[i,0] = random.uniform(0, w-1)
            pts[i,1] = random.uniform(0, h-1)

        rect = (0.0, 0.0, float(w), float(h))
        pts_relaxed = lloyd_relaxation(pts, rect, iters)
        facets = subdiv_voronoi_facets(pts_relaxed, rect)
        mosaic = build_shuffled_mosaic(self.img, facets, rect)
        self.result = mosaic
        self.show_image(mosaic)
        try:
            Image.fromarray(mosaic).save("mosaic_shuffled_result.png")
        except Exception:
            pass

    def save_to_desktop(self):
        if self.result is None:
            messagebox.showwarning("Внимание", "Нет результата для сохранения.")
            return
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        path = os.path.join(desktop, "mosaic_lloyd_shuffled.png")
        try:
            Image.fromarray(self.result).save(path)
            messagebox.showinfo("Сохранено", f"Сохранено на рабочий стол:\n{path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LloydShuffleApp(root)
    root.geometry("1100x700")
    root.mainloop()
