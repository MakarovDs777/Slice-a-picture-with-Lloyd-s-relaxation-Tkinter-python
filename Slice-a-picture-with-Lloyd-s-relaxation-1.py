import os
import random
import math
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2

# Параметры по умолчанию
DEFAULT_POINTS = 200
DEFAULT_LLOYD_ITERS = 3

def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось загрузить изображение:\n{e}")
        return None

def subdiv_voronoi_facets(points, rect):
    """
    Возвращает список фасетов (полигонов) для заданных points с помощью cv2.Subdiv2D.
    rect = (x0, y0, x1, y1)
    """
    x0, y0, x1, y1 = rect
    subdiv = cv2.Subdiv2D((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
    for p in points:
        # Subdiv2D любит tuple(float,float)
        subdiv.insert((float(p[0]), float(p[1])))

    # getVoronoiFacetList принимает список индексов; [] означает "все"
    try:
        facets, centers = subdiv.getVoronoiFacetList([])
    except Exception:
        # fallback signature (иногда возвращается три кортежа)
        try:
            ret, facets, centers = cv2.Subdiv2D.getVoronoiFacetList(subdiv, [])
        except Exception:
            return []

    # facets — список списков точек, возможно float. Обрежем полигон по прямоугольнику вручную позже, если нужно.
    poly_list = []
    for f in facets:
        if f is None or len(f) == 0:
            continue
        # f — ndarray Nx2
        pts = np.array(f, dtype=np.float64)
        poly_list.append(pts)
    return poly_list

def polygon_centroid(pts):
    """
    Геометрический центроид полигона (может быть вне целых координат).
    pts: Nx2 numpy array
    """
    if len(pts) < 3:
        return pts.mean(axis=0)
    x = pts[:,0]
    y = pts[:,1]
    a = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    A = a * 0.5
    if abs(A) < 1e-6:
        return pts.mean(axis=0)
    cx = np.sum((x + np.roll(x, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6.0 * A)
    cy = np.sum((y + np.roll(y, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)) / (6.0 * A)
    return np.array([cx, cy])

def clip_polygon_to_rect(pts, rect):
    """
    Простейшее отсечение полигона по прямоугольной области rect (x0,y0,x1,y1)
    Используем Sutherland–Hodgman (классический алгоритм отсечения).
    """
    def clip_edge(poly, edge):
        # edge: (a,b,c) полупространство ax+by<=c
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
                    # compute intersection prev->cur
                    dx = cur[0]-prev[0]; dy = cur[1]-prev[1]
                    denom = a*dx + b*dy
                    if abs(denom) > 1e-9:
                        t = (c - a*prev[0] - b*prev[1]) / denom
                        t = max(0.0, min(1.0, t))
                        inter = (prev[0] + dx*t, prev[1] + dy*t)
                        out.append(inter)
                out.append(tuple(cur))
            elif prev_inside:
                dx = cur[0]-prev[0]; dy = cur[1]-prev[1]
                denom = a*dx + b*dy
                if abs(denom) > 1e-9:
                    t = (c - a*prev[0] - b*prev[1]) / denom
                    t = max(0.0, min(1.0, t))
                    inter = (prev[0] + dx*t, prev[1] + dy*t)
                    out.append(inter)
            prev = cur
            prev_inside = cur_inside
        return out

    x0,y0,x1,y1 = rect
    # edges: left (x>=x0) -> -1*x + 0*y <= -x0  => a=-1,b=0,c=-x0  (we want x>=x0; so -x <= -x0)
    # but easier: use ax+by<=c form; to keep consistency we'll define edges to keep inside rect
    edges = [
        ( 1, 0, x1),  # x <= x1
        (-1, 0, -x0), # x >= x0  -> -x <= -x0
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
    w = x1 - x0; h = y1 - y0
    pts = points.copy()
    for _ in range(iters):
        facets = subdiv_voronoi_facets(pts, rect)
        new_pts = []
        for poly in facets:
            clipped = clip_polygon_to_rect(poly, rect)
            if len(clipped) == 0:
                continue
            c = polygon_centroid(clipped)
            # Если центроид вылез за границы, проецируем внутрь
            c[0] = min(max(c[0], x0), x1-1e-6)
            c[1] = min(max(c[1], y0), y1-1e-6)
            new_pts.append(c)
        if len(new_pts) == 0:
            break
        pts = np.array(new_pts, dtype=np.float64)
    return pts

def build_mosaic(image, points):
    """
    Для каждого полигона Вороного делаем маску и заливаем полигон средним цветом исходного изображения внутри маски.
    """
    h, w, _ = image.shape
    rect = (0.0, 0.0, float(w), float(h))
    facets = subdiv_voronoi_facets(points, rect)
    out = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)

    for poly in facets:
        clipped = clip_polygon_to_rect(poly, rect)
        if clipped is None or len(clipped) < 3:
            continue
        pts = np.array(clipped, dtype=np.int32)
        mask[:] = 0
        cv2.fillPoly(mask, [pts], 255)
        # cv2.mean возвращает среднее по каналам BGR, но у нас RGB — поэтому работаем с RGB напрямую
        # Но image — RGB; cv2.mean принимает image в BGR или RGB, оно просто считает среднее каналов.
        mean_color = cv2.mean(image, mask=mask)[:3]  # tuple of floats
        color = tuple(int(round(c)) for c in mean_color)
        # заполнить в выходном изображении
        cv2.fillPoly(out, [pts], color)
    return out

# ---------- GUI ----------
class LloydApp:
    def __init__(self, master):
        self.master = master
        master.title("Остроугольная мозаика — Lloyd + Voronoi")
        self.img = None
        self.display_imgtk = None

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

        tk.Button(control, text="Сгенерировать мозаику", command=self.generate).pack(fill="x", pady=8)
        tk.Button(control, text="Сохранить на рабочий стол", command=self.save_to_desktop).pack(fill="x", pady=4)

        # Канвас для показа изображения
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
        # Масштабируем для окна, но сохраняем оригинал для вычислений
        h, w, _ = arr.shape
        # размер канваса
        cw = max(300, min(w, 1000))
        ch = max(200, min(h, 800))
        # Подгоняем изображение под ширину канваса, но сохраняем соотношение
        # Для отображения используем PIL
        pil = Image.fromarray(arr)
        # Если картинка слишком большая, масштабируем для показа
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
        # случайные начальные точки
        pts = np.zeros((n,2), dtype=np.float64)
        for i in range(n):
            pts[i,0] = random.uniform(0, w-1)
            pts[i,1] = random.uniform(0, h-1)

        rect = (0.0, 0.0, float(w), float(h))
        # релаксация Ллойда по полигонам
        pts_relaxed = lloyd_relaxation(pts, rect, iters)
        # строим мозаику
        mosaic = build_mosaic(self.img, pts_relaxed)
        self.result = mosaic
        self.show_image(mosaic)
        # сохраняем автоматически в текущую папку под именем mosaic_result.png
        try:
            Image.fromarray(mosaic).save("mosaic_result.png")
        except Exception:
            pass

    def save_to_desktop(self):
        if not hasattr(self, "result") or self.result is None:
            messagebox.showwarning("Внимание", "Нет результата для сохранения.")
            return
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        path = os.path.join(desktop, "mosaic_lloyd_voronoi.png")
        try:
            Image.fromarray(self.result).save(path)
            messagebox.showinfo("Сохранено", f"Сохранено на рабочий стол:\n{path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LloydApp(root)
    root.geometry("1100x700")
    root.mainloop()
