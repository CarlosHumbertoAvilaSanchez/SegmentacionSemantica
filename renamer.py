import os
import shutil


def renombrar_archivos(ruta_carpeta):
    archivos = os.listdir(ruta_carpeta)
    count = 950  # CAMBIAR
    for archivo in archivos:
        ruta_actual = os.path.join(ruta_carpeta, archivo)
        count += 1
        if os.path.isfile(ruta_actual):
            nombre, extension = os.path.splitext(archivo)

            prefijo = str(count).zfill(5)
            nuevo_nombre = prefijo + extension

            # Construir la ruta completa del nuevo archivo
            nueva_ruta = os.path.join(ruta_carpeta, nuevo_nombre)

            # Renombrar el archivo
            shutil.move(ruta_actual, nueva_ruta)
            print(f"Archivo renombrado: '{archivo}' -> '{nuevo_nombre}'")


ruta_carpeta = "images/"
renombrar_archivos(ruta_carpeta)
