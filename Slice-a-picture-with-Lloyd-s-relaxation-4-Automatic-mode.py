import os
import random
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2

# Параметры миниатюр и значения по умолчанию
THUMB_W, THUMB_H = 140, 100
DEFAULT_POINTS = 200
DEFAULT_LLOYD_ITERS = 3
DEFAULT_BLUR_RADIUS = 12.0  # по умолчанию достаточно большой, чтобы получить мягкий тон

# ------------------ Вспомогательные функции (Voronoi / Lloyd / обрезка полигона) ------------------
def subdiv_voronoi_facets(points, rect):
    x0, y0, x1, y1 = rect
    # Subdiv2D требует int-диапазон в конструкторе: (x,y,w,h) — используем целые
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

# ------------------ Пастер с сильным feather (без чётких границ) ------------------
def paste_with_strong_feather(out, src_patch, tgt_mask, tx0, ty0, tx1, ty1, feather_radius):
    """
    Вставляет src_patch в out в bbox (tx0,ty0,tx1,ty1) с сильным feather:
    - расширение bbox (pad) чтобы blending выходил за края;
    - distance transform внутри маски -> alpha;
    - Gaussian blur по alpha;
    - небольшое размытие патча (low-pass) чтобы убрать острые детали.
    feather_radius: радиус (пиксели) — чем больше, тем мягче тон.
    """
    h_img, w_img, _ = out.shape
    tw = tx1 - tx0
    th = ty1 - ty0
    if tw <= 0 or th <= 0:
        return

    # pad — запас вокруг bbox, чтобы плавный переход выходил за границы клетки
    pad = max(2, int(round(feather_radius * 1.5)))
    ex_x0 = max(0, tx0 - pad)
    ex_y0 = max(0, ty0 - pad)
    ex_x1 = min(w_img, tx1 + pad)
    ex_y1 = min(h_img, ty1 + pad)
    ext_w = ex_x1 - ex_x0
    ext_h = ex_y1 - ex_y0

    if ext_w <= 0 or ext_h <= 0:
        return

    # ресайзим патч на расширенную область — INTER_LINEAR даёт более мягкое масштабирование
    resized_ext = cv2.resize(src_patch, (ext_w, ext_h), interpolation=cv2.INTER_LINEAR)

    # маска: ресайз на расширенную область (nearest + бинааризация)
    mask_ext = cv2.resize(tgt_mask, (ext_w, ext_h), interpolation=cv2.INTER_NEAREST)
    _, mask_ext = cv2.threshold(mask_ext, 127, 255, cv2.THRESH_BINARY)

    if feather_radius <= 0.5:
        # простая вставка (без feather)
        mask_bool = (mask_ext == 255)
        dst = out[ex_y0:ex_y1, ex_x0:ex_x1]
        dst[mask_bool] = resized_ext[mask_bool]
        out[ex_y0:ex_y1, ex_x0:ex_x1] = dst
        return

    # distance transform внутри маски: расстояние до ближайшего нуля (границы)
    dist = cv2.distanceTransform(mask_ext, cv2.DIST_L2, 5).astype(np.float32)

    # Нормируем расстояния по feather_radius: внутри центр -> alpha близок к 1, у границы -> 0
    alpha = np.clip(dist / max(1.0, float(feather_radius)), 0.0, 1.0)

    # Gaussian blur по alpha для ещё более мягкого перехода
    kr = max(3, int(round(feather_radius)) // 2 * 2 + 1)
    if kr < 3:
        kr = 3
    alpha = cv2.GaussianBlur(alpha, (kr, kr), sigmaX=max(0.1, feather_radius/3.0))

    # Слегка размываем сам патч, чтобы убрать острые углы/детали
    sigma_patch = max(0.3, feather_radius / 4.0)
    kpatch = max(3, int(round(sigma_patch)) * 2 + 1)
    blurred_patch = cv2.GaussianBlur(resized_ext, (kpatch, kpatch), sigmaX=sigma_patch, sigmaY=sigma_patch)

    # Композитим в расширенной зоне
    alpha_3 = alpha[..., None].astype(np.float32)
    dst_region = out[ex_y0:ex_y1, ex_x0:ex_x1].astype(np.float32)
    src_region = blurred_patch.astype(np.float32)
    composed = dst_region * (1.0 - alpha_3) + src_region * alpha_3
    out[ex_y0:ex_y1, ex_x0:ex_x1] = np.clip(composed, 0, 255).astype(np.uint8)

# ------------------ Извлечение целевых ячеек (bbox + маска) ------------------
def extract_target_cells(image, facets, rect):
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

# ------------------ Построение мозайки: single / cross source ------------------
def build_shuffled_mosaic_single_source(image, facets, rect, blur=False, blur_radius=DEFAULT_BLUR_RADIUS):
    h, w, _ = image.shape
    targets = extract_target_cells(image, facets, rect)
    if len(targets) == 0:
        return image.copy()

    # извлекаем патчи-источники из той же картинки
    patches = []
    for t in targets:
        x0,y0,x1,y1 = t["bbox"]
        patch = image[y0:y1, x0:x1].copy()
        mask_bbox = t["mask_bbox"].copy()
        patches.append({"patch": patch, "mask_bbox": mask_bbox, "bbox": (x0,y0,x1,y1)})

    perm = np.random.permutation(len(targets))
    out = np.zeros_like(image)

    for i, tgt in enumerate(targets):
        src = patches[perm[i]]
        tx0, ty0, tx1, ty1 = tgt["bbox"]
        tw = tx1 - tx0; th = ty1 - ty0
        if tw <= 0 or th <= 0:
            continue

        if not blur:
            # жёсткая вставка: ресайз патча в точный bbox — INTER_NEAREST сохраняет геометрию
            resized = cv2.resize(src["patch"], (tw, th), interpolation=cv2.INTER_NEAREST)
            tgt_mask = tgt["mask_bbox"]
            if tgt_mask.shape != (th, tw):
                tgt_mask = cv2.resize(tgt_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
                _, tgt_mask = cv2.threshold(tgt_mask, 127, 255, cv2.THRESH_BINARY)
            mask_bool = (tgt_mask == 255)
            dst_region = out[ty0:ty1, tx0:tx1]
            dst_region[mask_bool] = resized[mask_bool]
            out[ty0:ty1, tx0:tx1] = dst_region
        else:
            # плавная вставка с сильным feather
            tgt_mask = tgt["mask_bbox"]
            # передаём оригинальный src_patch (чтобы paste сделала ресайз на расширенную область)
            paste_with_strong_feather(out, src["patch"], tgt_mask, tx0, ty0, tx1, ty1, blur_radius)

    # по пустым пикселям (если остались) возвращаем оригинал
    zero_mask = np.all(out == 0, axis=2)
    out[zero_mask] = image[zero_mask]
    return out

def build_shuffled_mosaic_cross_source(target_image, facets, rect, source_images, blur=False, blur_radius=DEFAULT_BLUR_RADIUS):
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
        src_img = random.choice(source_images)
        sh, sw, _ = src_img.shape
        crop_w = min(sw, max(1, tw))
        crop_h = min(sh, max(1, th))
        crop_w = max(1, int(random.uniform(0.5, 1.2) * crop_w))
        crop_h = max(1, int(random.uniform(0.5, 1.2) * crop_h))
        crop_w = min(crop_w, sw); crop_h = min(crop_h, sh)
        x0 = random.randint(0, sw - crop_w) if sw - crop_w > 0 else 0
        y0 = random.randint(0, sh - crop_h) if sh - crop_h > 0 else 0
        src_patch = src_img[y0:y0+crop_h, x0:x0+crop_w].copy()

        if not blur:
            resized = cv2.resize(src_patch, (tw, th), interpolation=cv2.INTER_NEAREST)
            tgt_mask = tgt["mask_bbox"]
            if tgt_mask.shape != (th, tw):
                tgt_mask = cv2.resize(tgt_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
                _, tgt_mask = cv2.threshold(tgt_mask, 127, 255, cv2.THRESH_BINARY)
            mask_bool = (tgt_mask == 255)
            dst_region = out[ty0:ty1, tx0:tx1]
            dst_region[mask_bool] = resized[mask_bool]
            out[ty0:ty1, tx0:tx1] = dst_region
        else:
            tgt_mask = tgt["mask_bbox"]
            paste_with_strong_feather(out, src_patch, tgt_mask, tx0, ty0, tx1, ty1, blur_radius)

    zero_mask = np.all(out == 0, axis=2)
    out[zero_mask] = target_image[zero_mask]
    return out

# ------------------ GUI: галерея + генерация (с кнопками Blur / NoBlur) + авто ------------------
class GalleryApp:
    def __init__(self, master):
        self.master = master
        master.title("Галерея — Мозаика (Sharp / Strong Blur)")
        self.images = []
        self.selected = set()
        self.result_images = {}
        self.result_meta = {}  # хранит мета-информацию, например blur flag

        # автозапуск поля
        self.auto_after_id = None
        self.auto_running = False

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

        tk.Label(ctrl, text="Feather blur radius (px):").pack(anchor="w", pady=(10,0))
        self.blur_scale = tk.Scale(ctrl, from_=0, to=80, orient="horizontal", resolution=1)
        self.blur_scale.set(int(DEFAULT_BLUR_RADIUS))
        self.blur_scale.pack(fill="x", pady=2)

        # две кнопки генерации: без и с блюром
        btn_frame = tk.Frame(ctrl)
        btn_frame.pack(fill="x", pady=(12,4))
        tk.Button(btn_frame, text="Generate (No blur)", command=lambda: self.generate(blur=False)).pack(side="left", expand=True, fill="x", padx=2)
        tk.Button(btn_frame, text="Generate (With blur)", command=lambda: self.generate(blur=True)).pack(side="left", expand=True, fill="x", padx=2)

        # --- Новые контролы для авто-генерации ---
        tk.Label(ctrl, text="Auto interval (ms):").pack(anchor="w", pady=(10,0))
        self.auto_entry = tk.Entry(ctrl); self.auto_entry.insert(0, "1000"); self.auto_entry.pack(fill="x", pady=2)
        # Заменяем один чекбокс на радиокнопки: Auto No blur / Auto With blur
        tk.Label(ctrl, text="Auto blur:").pack(anchor="w")
        self.auto_blur_mode = tk.StringVar(value="no")  # 'no' или 'with'
        rbf = tk.Frame(ctrl)
        rbf.pack(anchor="w")
        tk.Radiobutton(rbf, text="No blur", variable=self.auto_blur_mode, value="no").pack(side="left")
        tk.Radiobutton(rbf, text="With blur", variable=self.auto_blur_mode, value="with").pack(side="left")

        btn_frame2 = tk.Frame(ctrl)
        btn_frame2.pack(fill="x", pady=(8,4))
        self.auto_btn = tk.Button(btn_frame2, text="Start Auto", command=self.toggle_auto)
        self.auto_btn.pack(side="left", expand=True, fill="x", padx=2)
        tk.Button(btn_frame2, text="Stop Auto", command=self.stop_auto).pack(side="left", expand=True, fill="x", padx=2)
        # ------------------------------------------------

        tk.Button(ctrl, text="Save All (Desktop)", command=self.save_all).pack(fill="x", pady=6)

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
        self.thumb_widgets = []

        # правая часть: просмотр выбранного/результата
        preview_frame = tk.Frame(master, bd=2, relief="sunken")
        preview_frame.pack(side="right", fill="both", expand=True, padx=6, pady=6)
        self.preview_canvas = tk.Canvas(preview_frame, bg="grey")
        self.preview_canvas.pack(fill="both", expand=True)
        self.preview_imgtk = None
        self.preview_index = None

        # Обработка закрытия: остановим авто-таймер если запущен
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

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
                thumb = pil.copy()
                thumb.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
                thumb_tk = ImageTk.PhotoImage(thumb)
                idx = len(self.images)
                self.images.append({"path": p, "arr": arr, "thumb_tk": thumb_tk, "w": arr.shape[1], "h": arr.shape[0]})
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
    def generate(self, blur=False, silent=False):
        if not self.selected:
            if not silent:
                messagebox.showwarning("Внимание", "Выберите хотя бы одно изображение в галерее.")
            return
        try:
            n = int(self.entry_points.get())
            iters = int(self.entry_iters.get())
        except Exception:
            if not silent:
                messagebox.showerror("Ошибка", "Параметры должны быть целыми числами.")
            return
        if n <= 0 or iters < 0:
            if not silent:
                messagebox.showerror("Ошибка", "Неверные параметры.")
            return
        mode = self.mode_var.get()
        sel_indices = list(self.selected)
        blur_radius = float(self.blur_scale.get()) if blur else 0.0

        self.result_images = {}
        self.result_meta = {}
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
                mosaic = build_shuffled_mosaic_single_source(img, facets, rect, blur=blur, blur_radius=blur_radius)
            else:
                sources = [self.images[i]["arr"] for i in sel_indices]
                mosaic = build_shuffled_mosaic_cross_source(img, facets, rect, sources, blur=blur, blur_radius=blur_radius)
            self.result_images[idx] = mosaic
            self.result_meta[idx] = {"blur": bool(blur), "blur_radius": float(blur_radius)}

        first = sel_indices[0]
        self.show_preview(self.result_images[first])
        if not silent:
            st = f"With blur (r={blur_radius})" if blur else "No blur"
            messagebox.showinfo("Готово", f"Сгенерировано ({st}) для {len(sel_indices)} изображений. Результат показан для первого выбранного.")

    # ---------- Авто-генерация ----------
    def toggle_auto(self):
        if self.auto_running:
            self.stop_auto()
        else:
            self.start_auto()

    def start_auto(self):
        if not self.selected:
            messagebox.showwarning("Внимание", "Выберите хотя бы одно изображение для авто-генерации.")
            return
        try:
            interval = int(self.auto_entry.get())
        except Exception:
            messagebox.showerror("Ошибка", "Интервал должен быть целым миллисекунд.")
            return
        if interval < 100:
            messagebox.showerror("Ошибка", "Интервал должен быть >=100 ms.")
            return
        self.auto_running = True
        self.auto_btn.config(text="Running...")
        self._auto_step()

    def _auto_step(self):
        if not self.auto_running:
            return
        blur_flag = (self.auto_blur_mode.get() == "with")
        # silent=True, чтобы не показывать dialog при каждом проходе
        self.generate(blur=blur_flag, silent=True)
        try:
            interval = int(self.auto_entry.get())
        except:
            interval = 1000
        # запланируем следующий шаг
        self.auto_after_id = self.master.after(max(100, interval), self._auto_step)

    def stop_auto(self):
        if self.auto_after_id:
            try:
                self.master.after_cancel(self.auto_after_id)
            except Exception:
                pass
            self.auto_after_id = None
        self.auto_running = False
        self.auto_btn.config(text="Start Auto")

    # ---------- Save ----------
    def save_all(self):
        if not self.result_images:
            messagebox.showwarning("Внимание", "Нет результатов. Сначала нажмите Generate.")
            return
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        saved = 0
        for idx, arr in self.result_images.items():
            base = os.path.splitext(os.path.basename(self.images[idx]["path"]))[0]
            meta = self.result_meta.get(idx, {})
            suffix = "_blur" if meta.get("blur", False) else "_sharp"
            name = f"{base}{suffix}.png"
            path = os.path.join(desktop, name)
            try:
                Image.fromarray(arr).save(path)
                saved += 1
            except Exception as e:
                print("Save failed", path, e)
        messagebox.showinfo("Сохранено", f"Сохранены {saved} файлов на рабочий стол.")

    # ---------- Закрытие ----------
    def on_close(self):
        # остановим авто-таймер, если запущен
        self.stop_auto()
        self.master.destroy()

# ------------------ main ------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = GalleryApp(root)
    root.geometry("1300x800")
    root.mainloop()
