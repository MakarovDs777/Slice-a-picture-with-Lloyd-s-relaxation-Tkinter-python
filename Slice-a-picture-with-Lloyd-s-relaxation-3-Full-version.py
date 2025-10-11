import os
import random
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2

# Параметры
THUMB_W, THUMB_H = 140, 100
DEFAULT_POINTS = 200
DEFAULT_LLOYD_ITERS = 3

# ------------------ Вспомогательные функции (Voronoi/Lloyd/отсечение) ------------------
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

# ------------------ Построение мозайки (sharp, shuffle) ------------------
def extract_target_cells(image, facets, rect):
    """Возвращает список целевых ячеек (bbox, mask_bbox, poly_pts) для данного изображения"""
    h, w, _ = image.shape
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
    return targets

def build_shuffled_mosaic_single_source(image, facets, rect):
    """Патчи берутся из той же картинки (как раньше)."""
    h, w, _ = image.shape
    targets = extract_target_cells(image, facets, rect)
    n = len(targets)
    if n == 0:
        return image.copy()
    # извлекаем патчи-источники из той же картинки
    patches = []
    for t in targets:
        x0,y0,x1,y1 = t["bbox"]
        patch = image[y0:y1, x0:x1].copy()
        mask_bbox = t["mask_bbox"].copy()
        patches.append({"patch": patch, "mask_bbox": mask_bbox, "bbox": (x0,y0,x1,y1)})
    # перемешивание
    perm = np.random.permutation(n)
    out = np.zeros_like(image)
    for i, tgt in enumerate(targets):
        src = patches[perm[i]]
        tx0, ty0, tx1, ty1 = tgt["bbox"]
        tw = tx1 - tx0; th = ty1 - ty0
        if tw <= 0 or th <= 0:
            continue
        resized = cv2.resize(src["patch"], (tw, th), interpolation=cv2.INTER_NEAREST)
        tgt_mask = tgt["mask_bbox"]
        if tgt_mask.shape != (th, tw):
            tgt_mask = cv2.resize(tgt_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
            _, tgt_mask = cv2.threshold(tgt_mask, 127, 255, cv2.THRESH_BINARY)
        mask_bool = (tgt_mask == 255)
        dst_region = out[ty0:ty1, tx0:tx1]
        dst_region[mask_bool] = resized[mask_bool]
        out[ty0:ty1, tx0:tx1] = dst_region
    zero_mask = np.all(out == 0, axis=2)
    out[zero_mask] = image[zero_mask]
    return out

def build_shuffled_mosaic_cross_source(target_image, facets, rect, source_images):
    """
    Для каждой целевой ячейки берём случайный кусок (случайная позиция) из случайного source_image,
    масштабируем (nearest) в размеры bbox и вставляем по маске целевой ячейки.
    """
    h, w, _ = target_image.shape
    targets = extract_target_cells(target_image, facets, rect)
    if len(targets) == 0:
        return target_image.copy()
    out = np.zeros_like(target_image)
    for tgt in targets:
        tx0, ty0, tx1, ty1 = tgt["bbox"]
        tw = tx1 - tx0; th = ty1 - ty0
        if tw <= 0 or th <= 0:
            continue
        # выбрать случайный источник
        src_img = random.choice(source_images)
        sh, sw, _ = src_img.shape
        # выбрать случайный crop в source размером примерно tw x th (если больше размера -> уменьшать)
        # Чтобы не выходить за границы:
        crop_w = min(sw, tw)
        crop_h = min(sh, th)
        # иногда берем чуть более широкий crop чтобы потом ресайз давал вариативность
        crop_w = max(1, int(random.uniform(0.5, 1.2) * crop_w))
        crop_h = max(1, int(random.uniform(0.5, 1.2) * crop_h))
        crop_w = min(crop_w, sw); crop_h = min(crop_h, sh)
        x0 = random.randint(0, sw - crop_w) if sw - crop_w > 0 else 0
        y0 = random.randint(0, sh - crop_h) if sh - crop_h > 0 else 0
        src_patch = src_img[y0:y0+crop_h, x0:x0+crop_w].copy()
        resized = cv2.resize(src_patch, (tw, th), interpolation=cv2.INTER_NEAREST)
        tgt_mask = tgt["mask_bbox"]
        if tgt_mask.shape != (th, tw):
            tgt_mask = cv2.resize(tgt_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
            _, tgt_mask = cv2.threshold(tgt_mask, 127, 255, cv2.THRESH_BINARY)
        mask_bool = (tgt_mask == 255)
        dst_region = out[ty0:ty1, tx0:tx1]
        dst_region[mask_bool] = resized[mask_bool]
        out[ty0:ty1, tx0:tx1] = dst_region
    zero_mask = np.all(out == 0, axis=2)
    out[zero_mask] = target_image[zero_mask]
    return out

# ------------------ GUI: галерея + генерация ------------------
class GalleryApp:
    def __init__(self, master):
        self.master = master
        master.title("Галерея — Мозаика с перемещёнными фрагментами")
        self.images = []         # list of dict: {"path", "arr", "thumb_tk", "w","h"}
        self.selected = set()    # индексы выбранных миниатюр
        self.result_images = {}  # результаты по индексам (numpy arrays)

        # левая панель управления
        ctrl = tk.Frame(master, padx=6, pady=6)
        ctrl.pack(side="left", fill="y")

        tk.Button(ctrl, text="Add images", command=self.add_images).pack(fill="x", pady=3)
        tk.Button(ctrl, text="Clear gallery", command=self.clear_gallery).pack(fill="x", pady=3)
        tk.Button(ctrl, text="Select All", command=self.select_all).pack(fill="x", pady=3)
        tk.Button(ctrl, text="Deselect All", command=self.deselect_all).pack(fill="x", pady=3)

        tk.Label(ctrl, text="Число точек:").pack(anchor="w", pady=(10,0))
        self.entry_points = tk.Entry(ctrl); self.entry_points.insert(0, str(DEFAULT_POINTS)); self.entry_points.pack(fill="x", pady=2)
        tk.Label(ctrl, text="Итераций Ллойда:").pack(anchor="w")
        self.entry_iters = tk.Entry(ctrl); self.entry_iters.insert(0, str(DEFAULT_LLOYD_ITERS)); self.entry_iters.pack(fill="x", pady=2)

        tk.Label(ctrl, text="Режим:").pack(anchor="w", pady=(10,0))
        self.mode_var = tk.StringVar(value="separate")
        tk.Radiobutton(ctrl, text="Separate (свои патчи)", variable=self.mode_var, value="separate").pack(anchor="w")
        tk.Radiobutton(ctrl, text="Cross-source (пулы из выбранных)", variable=self.mode_var, value="cross").pack(anchor="w")

        tk.Button(ctrl, text="Generate", command=self.generate).pack(fill="x", pady=(12,4))
        tk.Button(ctrl, text="Save All (Desktop)", command=self.save_all).pack(fill="x", pady=3)

        # центральная часть: галерея (скроллируемый канвас)
        gallery_frame = tk.Frame(master, bd=2, relief="sunken")
        gallery_frame.pack(side="left", fill="both", expand=False, padx=6, pady=6)
        self.canvas = tk.Canvas(gallery_frame, width=THUMB_W+20, bg="#f0f0f0")
        self.v_scroll = tk.Scrollbar(gallery_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set)
        self.v_scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.inner_frame = tk.Frame(self.canvas, bg="#f0f0f0")
        self.canvas.create_window((0,0), window=self.inner_frame, anchor="nw")
        self.thumb_widgets = []  # for drawing borders and binding

        # правая часть: просмотр выбранного/результата
        preview_frame = tk.Frame(master, bd=2, relief="sunken")
        preview_frame.pack(side="right", fill="both", expand=True, padx=6, pady=6)
        self.preview_canvas = tk.Canvas(preview_frame, bg="grey")
        self.preview_canvas.pack(fill="both", expand=True)
        self.preview_imgtk = None
        self.preview_index = None

    # ---------- Галерея ----------
    def add_images(self):
        paths = filedialog.askopenfilenames(title="Выберите изображения",
                                            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if not paths:
            return
        for p in paths:
            try:
                pil = Image.open(p).convert("RGB")
                arr = np.array(pil)
                w, h = arr.shape[1], arr.shape[0]
                thumb = pil.copy()
                thumb.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
                thumb_tk = ImageTk.PhotoImage(thumb)
                idx = len(self.images)
                self.images.append({"path": p, "arr": arr, "thumb_tk": thumb_tk, "w": w, "h": h})
                self._add_thumb_widget(idx)
            except Exception as e:
                print("Failed load", p, e)
        self._refresh_gallery()

    def _add_thumb_widget(self, idx):
        frame = tk.Frame(self.inner_frame, bd=2, relief="flat", bg="#ffffff")
        label = tk.Label(frame, image=self.images[idx]["thumb_tk"], bg="#fff")
        label.pack()
        info = tk.Label(frame, text=os.path.basename(self.images[idx]["path"]), wraplength=THUMB_W, bg="#fff", font=("Arial", 8))
        info.pack()
        frame.grid(row=idx, column=0, padx=4, pady=4, sticky="nw")
        frame.bind("<Button-1>", lambda e, i=idx: self.toggle_select(i))
        label.bind("<Button-1>", lambda e, i=idx: self.toggle_select(i))
        info.bind("<Button-1>", lambda e, i=idx: self.toggle_select(i))
        self.thumb_widgets.append(frame)

    def _refresh_gallery(self):
        # обновить бордеры по selection
        for i, w in enumerate(self.thumb_widgets):
            if i in self.selected:
                w.config(bd=3, relief="solid", bg="#cfefff")
            else:
                w.config(bd=2, relief="flat", bg="#fff")
        self.inner_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def clear_gallery(self):
        self.images = []
        self.selected = set()
        for w in self.thumb_widgets:
            w.destroy()
        self.thumb_widgets = []
        self._refresh_gallery()
        self.preview_canvas.delete("all")
        self.preview_imgtk = None

    def select_all(self):
        self.selected = set(range(len(self.images)))
        self._refresh_gallery()

    def deselect_all(self):
        self.selected = set()
        self._refresh_gallery()

    def toggle_select(self, idx):
        if idx in self.selected:
            self.selected.remove(idx)
        else:
            self.selected.add(idx)
        # показать в preview при одиночном клике
        self.preview_index = idx
        self.show_preview(self.images[idx]["arr"])
        self._refresh_gallery()

    # ---------- Preview ----------
    def show_preview(self, arr):
        pil = Image.fromarray(arr)
        max_display = 900
        scale = min(1.0, max_display / max(pil.width, pil.height))
        if scale < 1.0:
            pil = pil.resize((int(pil.width*scale), int(pil.height*scale)), Image.LANCZOS)
        self.preview_imgtk = ImageTk.PhotoImage(pil)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, anchor="nw", image=self.preview_imgtk)
        self.preview_canvas.config(width=self.preview_imgtk.width(), height=self.preview_imgtk.height())

    # ---------- Генерация ----------
    def generate(self):
        if not self.selected:
            messagebox.showwarning("Внимание", "Выберите хотя бы одно изображение в галерее.")
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
        mode = self.mode_var.get()
        sel_indices = list(self.selected)
        # подготовка: для каждого выбранного изображения делаем relaxed points и facets
        self.result_images = {}
        for idx in sel_indices:
            img = self.images[idx]["arr"]
            h, w = img.shape[0], img.shape[1]
            pts = np.zeros((n,2), dtype=np.float64)
            for i in range(n):
                pts[i,0] = random.uniform(0, w-1)
                pts[i,1] = random.uniform(0, h-1)
            rect = (0.0, 0.0, float(w), float(h))
            pts_relaxed = lloyd_relaxation(pts, rect, iters)
            facets = subdiv_voronoi_facets(pts_relaxed, rect)
            if mode == "separate":
                mosaic = build_shuffled_mosaic_single_source(img, facets, rect)
            else:  # cross-source
                # соберём пул source_images (numpy arrays) — все выбранные картинки
                sources = [self.images[i]["arr"] for i in sel_indices]
                mosaic = build_shuffled_mosaic_cross_source(img, facets, rect, sources)
            self.result_images[idx] = mosaic
        # показать первый результат
        first = sel_indices[0]
        self.show_preview(self.result_images[first])
        messagebox.showinfo("Готово", f"Сгенерировано для {len(sel_indices)} изображений. Результат показан для первого выбранного.")

    def save_all(self):
        if not self.result_images:
            messagebox.showwarning("Внимание", "Нет результатов. Сначала нажмите Generate.")
            return
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        for idx, arr in self.result_images.items():
            base = os.path.splitext(os.path.basename(self.images[idx]["path"]))[0]
            name = f"{base}_mosaic.png"
            path = os.path.join(desktop, name)
            try:
                Image.fromarray(arr).save(path)
            except Exception as e:
                print("Save failed", path, e)
        messagebox.showinfo("Сохранено", f"Сохранены {len(self.result_images)} файлов на рабочий стол.")

# ------------------ main ------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = GalleryApp(root)
    root.geometry("1300x800")
    root.mainloop()
