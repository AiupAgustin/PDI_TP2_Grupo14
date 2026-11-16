import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

RUTA_IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagenes", "monedas.jpg")


def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)


def imfillhole(img):
    img_flood_fill = img.copy().astype("uint8")
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(img_flood_fill, mask, (0, 0), 255)
    img_flood_fill_inv = cv2.bitwise_not(img_flood_fill)
    return img | img_flood_fill_inv


def factor_forma(cnt):
    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)
    return area, perimetro, area / perimetro**2


K_MEDIANA = 9
CANNY_TH = (20, 60)
K_CLAUSURA = 25
AREA_MIN_OBJ = 20000
K_APERTURA = 51
INV_FP_UMBRAL = 15.0


def punto_a(img_rgb, mostrar=True):
    print("\n========== A. Segmentación de monedas y dados ==========")
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    gray_f = cv2.medianBlur(gray, K_MEDIANA)
    bordes = cv2.Canny(gray_f, *CANNY_TH)

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K_CLAUSURA, K_CLAUSURA))
    bordes_c = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, se)

    cnts, _ = cv2.findContours(bordes_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    relleno = np.zeros_like(gray)
    for c in cnts:
        hull = cv2.convexHull(c)
        if cv2.contourArea(hull) >= AREA_MIN_OBJ:
            cv2.drawContours(relleno, [hull], -1, 255, -1)

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K_APERTURA, K_APERTURA))
    mascara = cv2.morphologyEx(relleno, cv2.MORPH_OPEN, se)

    objetos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(f"[A] Objetos segmentados: {len(objetos)}")

    if mostrar:
        plt.figure(figsize=(14, 8))
        plt.suptitle("A. Segmentación de monedas y dados")
        ax1 = plt.subplot(231); imshow(img_rgb, new_fig=False, color_img=True, title="Original")
        plt.subplot(232, sharex=ax1, sharey=ax1); imshow(gray_f, new_fig=False, title=f"Grises + mediana {K_MEDIANA}x{K_MEDIANA}")
        plt.subplot(233, sharex=ax1, sharey=ax1); imshow(bordes, new_fig=False, title=f"Canny {CANNY_TH}")
        plt.subplot(234, sharex=ax1, sharey=ax1); imshow(bordes_c, new_fig=False, title=f"Clausura {K_CLAUSURA}x{K_CLAUSURA}")
        plt.subplot(235, sharex=ax1, sharey=ax1); imshow(relleno, new_fig=False, title="Relleno (envolvente convexa)")
        plt.subplot(236, sharex=ax1, sharey=ax1); imshow(mascara, new_fig=False, title=f"Apertura {K_APERTURA}x{K_APERTURA} - {len(objetos)} objetos")
        plt.tight_layout()
        plt.show(block=False)

    monedas, dados = [], []
    print("\n[A] Factor de forma de cada objeto (círculo ideal: 1/Fp = 12.57 / cuadrado: 1/Fp = 16)")
    print(f"     {'#':>3} {'x':>5} {'y':>5} {'Área':>9} {'Perím.':>8} {'Fp':>7} {'1/Fp':>6}  Clase")
    for i, c in enumerate(objetos):
        area, per, fp = factor_forma(c)
        x, y, w, h = cv2.boundingRect(c)
        clase = "moneda" if 1 / fp < INV_FP_UMBRAL else "dado"
        (monedas if clase == "moneda" else dados).append(c)
        print(f"     {i:3d} {x:5d} {y:5d} {area:9.0f} {per:8.1f} {fp:7.4f} {1/fp:6.2f}  {clase}")
    print(f"[A] Monedas: {len(monedas)} - Dados: {len(dados)}")

    if mostrar:
        out = img_rgb.copy()
        for c in monedas:
            cv2.drawContours(out, [c], -1, (255, 0, 0), 12)
        for c in dados:
            cv2.drawContours(out, [c], -1, (0, 255, 0), 12)
        inv_fp = sorted([(1 / factor_forma(c)[2]) for c in objetos])
        plt.figure(figsize=(14, 6))
        plt.suptitle("A. Separación por factor de forma")
        plt.subplot(121); imshow(out, new_fig=False, color_img=True, title=f"Monedas (rojo): {len(monedas)} - Dados (verde): {len(dados)}")
        plt.subplot(122)
        plt.plot(inv_fp, 'o')
        plt.axhline(INV_FP_UMBRAL, color='r', linestyle='--', label=f"Umbral 1/Fp = {INV_FP_UMBRAL}")
        plt.axhline(4 * np.pi, color='b', linestyle=':', label="Círculo ideal (12.57)")
        plt.axhline(16, color='g', linestyle=':', label="Cuadrado ideal (16)")
        plt.title("1/Fp de cada objeto (ordenado)"), plt.xlabel("Objeto"), plt.ylabel("1/Fp"), plt.legend(), plt.grid(True)
        plt.tight_layout()
        plt.show(block=False)

    return gray, monedas, dados


AREA_10C_MAX = 78000
AREA_1P_MAX = 100000
TIPOS_MONEDA = {"10 centavos": 0.10, "1 peso": 1.00, "50 centavos": 0.50}
COLORES_MONEDA = {"10 centavos": (0, 120, 255), "1 peso": (255, 0, 0), "50 centavos": (0, 200, 0)}


def punto_b(img_rgb, monedas, mostrar=True):
    print("\n========== B. Clasificación y conteo de monedas ==========")
    tipos = []
    for c in monedas:
        area = cv2.contourArea(c)
        if area < AREA_10C_MAX:
            tipos.append("10 centavos")
        elif area < AREA_1P_MAX:
            tipos.append("1 peso")
        else:
            tipos.append("50 centavos")

    total = 0.0
    for t, valor in TIPOS_MONEDA.items():
        n = tipos.count(t)
        total += n * valor
        print(f"     {t:12s}: {n:2d}")
    print(f"     {'TOTAL':12s}: {len(monedas):2d} monedas  -->  $ {total:.2f}")

    if mostrar:
        out = img_rgb.copy()
        for c, t in zip(monedas, tipos):
            x, y, w, h = cv2.boundingRect(c)
            cv2.drawContours(out, [c], -1, COLORES_MONEDA[t], 14)
            cv2.putText(out, t, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 2.2, COLORES_MONEDA[t], 7)
        areas = [cv2.contourArea(c) for c in monedas]
        plt.figure(figsize=(14, 6))
        plt.suptitle("B. Clasificación de monedas")
        plt.subplot(121)
        titulo = " / ".join(f"{t}: {tipos.count(t)}" for t in TIPOS_MONEDA)
        imshow(out, new_fig=False, color_img=True, title=titulo)
        plt.subplot(122)
        for t in TIPOS_MONEDA:
            a = [ar for ar, tt in zip(areas, tipos) if tt == t]
            plt.plot(a, np.zeros(len(a)) + list(TIPOS_MONEDA).index(t), 'o', color=np.array(COLORES_MONEDA[t]) / 255, label=t)
        plt.axvline(AREA_10C_MAX, color='k', linestyle='--')
        plt.axvline(AREA_1P_MAX, color='k', linestyle='--')
        plt.yticks(range(3), list(TIPOS_MONEDA)), plt.xlabel("Área [px]"), plt.title("Áreas y umbrales"), plt.grid(True)
        plt.tight_layout()
        plt.show(block=False)
    return tipos


PUNTO_AREA_MIN_REL = 0.015
PUNTO_INV_FP_MAX = 20.0


def punto_c(gray, dados, mostrar=True):
    print("\n========== C. Valor de la cara superior de cada dado ==========")
    valores = []
    for i, c in enumerate(dados):
        x, y, w, h = cv2.boundingRect(c)
        crop = gray[y:y + h, x:x + w]
        mask = np.zeros_like(crop)
        cv2.drawContours(mask, [c - [x, y]], -1, 255, -1)
        area_dado = cv2.contourArea(c)

        _, th = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        th = cv2.bitwise_and(th, mask)
        th_op = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        th_fh = imfillhole(th_op)
        cnts, _ = cv2.findContours(th_fh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        puntos = []
        for p in cnts:
            a = cv2.contourArea(p)
            if a < PUNTO_AREA_MIN_REL * area_dado:
                continue
            _, _, fp = factor_forma(p)
            if 1 / fp <= PUNTO_INV_FP_MAX:
                puntos.append(p)
        valores.append(len(puntos))
        print(f"     Dado {i + 1} (x={x}, y={y}): {len(cnts)} regiones oscuras --> {len(puntos)} puntos en la cara superior --> VALOR = {len(puntos)}")

        if mostrar:
            res = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(res, cnts, -1, (255, 0, 0), 3)
            cv2.drawContours(res, puntos, -1, (0, 255, 0), 5)
            plt.figure(figsize=(14, 4))
            plt.suptitle(f"C. Dado {i + 1}")
            plt.subplot(141); imshow(crop, new_fig=False, title="Recorte")
            plt.subplot(142); imshow(th, new_fig=False, title="Otsu invertido (dentro del dado)")
            plt.subplot(143); imshow(th_fh, new_fig=False, title="Apertura + relleno de huecos")
            plt.subplot(144); imshow(res, new_fig=False, color_img=True, title=f"Puntos (verde): {len(puntos)}")
            plt.tight_layout()
            plt.show(block=False)

    print(f"[C] Dados: {len(dados)} - Valores: {valores} - Suma: {sum(valores)}")
    return valores


def resultado_final(img_rgb, monedas, tipos, dados, valores):
    out = img_rgb.copy()
    for c, t in zip(monedas, tipos):
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), COLORES_MONEDA[t], 12)
        cv2.putText(out, t, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 2.2, COLORES_MONEDA[t], 7)
    for c, v in zip(dados, valores):
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 255), 12)
        cv2.putText(out, f"Dado: {v}", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 255), 8)
    conteo = ", ".join(f"{tipos.count(t)} x {t}" for t in TIPOS_MONEDA)
    total = sum(TIPOS_MONEDA[t] for t in tipos)
    plt.figure(figsize=(12, 8))
    imshow(out, new_fig=False, color_img=True,
           title=f"Monedas: {conteo} (total $ {total:.2f})\nDados: {len(dados)} - valores {valores} (suma {sum(valores)})")
    plt.tight_layout()
    plt.show(block=True)


def main(mostrar=True):
    img = cv2.imread(RUTA_IMG, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(RUTA_IMG)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Imagen: {RUTA_IMG} - {img.shape}")

    gray, monedas, dados = punto_a(img_rgb, mostrar)
    tipos = punto_b(img_rgb, monedas, mostrar)
    valores = punto_c(gray, dados, mostrar)
    if mostrar:
        resultado_final(img_rgb, monedas, tipos, dados, valores)
    return tipos, valores


if __name__ == "__main__":
    main()
