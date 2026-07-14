import gradio as gr
import cv2
import numpy as np

# FUNKCJE

def prepare(image, gray_mode):
    if image is None:
        return None
    if gray_mode:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def liniowe_przeksztalcenie(image, a, b, gray_mode):
    img = prepare(image, gray_mode)
    processed = np.clip(img.astype(float) * a + b, 0, 255).astype(np.uint8)
    return img, processed


def wygladzanie_splotem(image, rozmiar, gauss, gray_mode):
    img = prepare(image, gray_mode)

    if gauss:
        processed = cv2.GaussianBlur(img, (rozmiar, rozmiar), 0)
    else:
        kernel = np.ones((rozmiar, rozmiar), np.float32) / (rozmiar * rozmiar)
        processed = cv2.filter2D(img, -1, kernel)

    return img, processed


def gradient_obrazu(image, gray_mode):
    img = prepare(image, True)

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    mag = np.sqrt(sobelx**2 + sobely**2)
    processed = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return img, processed


def sobel(image, gray_mode):
    img = prepare(image, True)

    sx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sy = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    mag = cv2.magnitude(sx, sy)
    processed = cv2.convertScaleAbs(mag)

    return img, processed


def laplace(image, gray_mode):
    img = prepare(image, True)

    lap = cv2.Laplacian(img, cv2.CV_64F)
    processed = cv2.convertScaleAbs(lap)

    return img, processed


def translacja(image, tx, ty, gray_mode):
    img = prepare(image, gray_mode)

    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])

    processed = cv2.warpAffine(img, M, (cols, rows))
    return img, processed


def skalowanie(image, fx, fy, gray_mode):
    img = prepare(image, gray_mode)

    processed = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    return img, processed


def obrot(image, kat, gray_mode):
    img = prepare(image, gray_mode)

    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), kat, 1)

    processed = cv2.warpAffine(img, M, (cols, rows))
    return img, processed


# UI

with gr.Blocks(title="Praca dyplomowa – Damian KOSIŃSKI", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
# Zastosowanie analizy matematycznej w przetwarzaniu obrazów  
**Damian KOSIŃSKI**  
Uniwersytet w Białymstoku • 2026  
""")

    # ===== 3.1 =====
    with gr.Tab("3.1 Jasność / kontrast"):
        img = gr.Image(type="numpy")
        orig = gr.Image(label="Oryginał")
        out = gr.Image(label="Wynik")

        a = gr.Slider(0.1, 3.0, 1.5, label="Kontrast (a)")
        b = gr.Slider(-50, 50, 20, label="Jasność (b)")
        gray = gr.Checkbox(True, label="Tryb szarości")

        gr.Button("Zastosuj").click(liniowe_przeksztalcenie,
                                   [img, a, b, gray],
                                   [orig, out])

    # ===== 3.2 =====
    with gr.Tab("3.2 Splot"):
        img = gr.Image(type="numpy")
        orig = gr.Image()
        out = gr.Image()

        k = gr.Slider(3, 15, 5, step=2)
        gauss = gr.Checkbox(True, label="Gauss")
        gray = gr.Checkbox(True, label="Tryb szarości")

        gr.Button("Filtruj").click(wygladzanie_splotem,
                                  [img, k, gauss, gray],
                                  [orig, out])

    # ===== 3.3 =====
    with gr.Tab("3.3 Gradient"):
        img = gr.Image(type="numpy")
        orig = gr.Image()
        out = gr.Image()


        gr.Button("Gradient").click(gradient_obrazu,
                                   [img, gray],
                                   [orig, out])

    # SOBEL
    with gr.Tab("3.4 Sobel"):
        img = gr.Image(type="numpy")
        orig = gr.Image()
        out = gr.Image()


        gr.Button("Sobel").click(sobel,
                                [img, gray],
                                [orig, out])

    # LAPLACE
    with gr.Tab("3.4 Laplace"):
        img = gr.Image(type="numpy")
        orig = gr.Image()
        out = gr.Image()


        gr.Button("Laplace").click(laplace,
                                  [img, gray],
                                  [orig, out])

    with gr.Tab("3.5.1 Translacja"):
        img = gr.Image(type="numpy")
        orig = gr.Image()
        out = gr.Image()

        tx = gr.Slider(-200, 200, 50)
        ty = gr.Slider(-200, 200, 30)
        gray = gr.Checkbox(True, label="Tryb szarości")

        gr.Button("Translacja").click(translacja,
                                     [img, tx, ty, gray],
                                     [orig, out])

    with gr.Tab("3.5.2 Skalowanie"):
        img = gr.Image(type="numpy")
        orig = gr.Image()
        out = gr.Image()

        fx = gr.Slider(0.5, 3.0, 1.5)
        fy = gr.Slider(0.5, 3.0, 1.5)
        gray = gr.Checkbox(True, label="Tryb szarości")

        gr.Button("Skaluj").click(skalowanie,
                                 [img, fx, fy, gray],
                                 [orig, out])

    with gr.Tab("3.5.3 Obrót"):
        img = gr.Image(type="numpy")
        orig = gr.Image()
        out = gr.Image()

        kat = gr.Slider(-180, 180, 45)
        gray = gr.Checkbox(True, label="Tryb szarości")

        gr.Button("Obrót").click(obrot,
                                [img, kat, gray],
                                [orig, out])


demo.launch()