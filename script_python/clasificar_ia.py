import rasterio
import numpy as np
import geopandas as gpd
from rasterio.mask import mask
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time
from rasterio.windows import Window

SHAPEFILE_MUESTRAS = r'D:/Land Surveyor/Ejercicios QGIS/Clasificación de Uso del Suelo IA/Muestras.gpkg'
IMAGEN_SATELITAL_BASE = r'D:/Land Surveyor/Ejercicios QGIS/Clasificación de Uso del Suelo IA/Virtual_rast.tif'
IMAGEN_ENRIQUECIDA = r'D:/Land Surveyor/Ejercicios QGIS/Clasificación de Uso del Suelo IA/imagen_enriquecida_indices.tif'
IMAGEN_CLASIFICADA_SALIDA = r'D:/Land Surveyor/Ejercicios QGIS/Clasificación de Uso del Suelo IA/mapa_clasificado_avanzado.tif'

print("Paso 1: Creando imagen enriquecida con múltiples bandas e índices...")
with rasterio.open(IMAGEN_SATELITAL_BASE) as src:
    profile = src.profile.copy()
    verde = src.read(1).astype(float)
    rojo = src.read(2).astype(float)
    nir = src.read(3).astype(float)
    swir = src.read(4).astype(float)
    
    np.seterr(divide='ignore', invalid='ignore')
    
    ndvi = (nir - rojo) / (nir + rojo)
    ndwi = (verde - nir) / (verde + nir)
    ndbi = (swir - nir) / (swir + nir)
    
    indices = [ndvi, ndwi, ndbi]
    for i in indices:
        i[np.isnan(i)] = 0
        
    todas_las_bandas = [src.read(i+1) for i in range(src.count)] + indices
    
    profile.update(count=len(todas_las_bandas), dtype='float32')
    
    with rasterio.open(IMAGEN_ENRIQUECIDA, 'w', **profile) as dst:
        for i, banda in enumerate(todas_las_bandas):
            dst.write(banda.astype('float32'), i + 1)
print("Imagen enriquecida creada en:", IMAGEN_ENRIQUECIDA)

print("\nPaso 2: Cargando muestras y extrayendo píxeles de la imagen enriquecida...")
muestras = gpd.read_file(SHAPEFILE_MUESTRAS)
X = []
y = []

with rasterio.open(IMAGEN_ENRIQUECIDA) as src:
    for index, row in muestras.iterrows():
        clase_id = row['class_id']
        out_image, _ = mask(src, [row.geometry], crop=True, all_touched=True, nodata=0)
        out_image_reshaped = out_image.reshape(out_image.shape[0], -1).T
        out_image_filtered = out_image_reshaped[~np.all(out_image_reshaped == 0, axis=1)]
        X.extend(out_image_filtered)
        y.extend([clase_id] * len(out_image_filtered))

X = np.array(X)
y = np.array(y)

print("\nEvaluando el modelo...")
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)
modelo_evaluacion = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
modelo_evaluacion.fit(X_entrenamiento, y_entrenamiento)
predicciones = modelo_evaluacion.predict(X_prueba)
informe_texto = classification_report(y_prueba, predicciones)

print("\nPaso 3: Entrenando modelo final con TODOS los datos...")
modelo_final = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
modelo_final.fit(X, y)
print("Paso 4 y 5: Clasificando la imagen enriquecida por bloques...")

with rasterio.open(IMAGEN_ENRIQUECIDA) as src:
    profile = src.profile.copy()
    profile.update(dtype='uint8', count=1, nodata=0)
    
    with rasterio.open(IMAGEN_CLASIFICADA_SALIDA, 'w', **profile) as dst:
        for ji, window in src.block_windows(1):
            print(f"Procesando bloque {ji}... ", end='')
            img_block = src.read(window=window)
            img_transpuesta = np.transpose(img_block, (1, 2, 0))
            img_aplanada = img_transpuesta.reshape(-1, src.count)
            clasificacion_bloque = modelo_final.predict(img_aplanada)
            mapa_clasificado_bloque = clasificacion_bloque.reshape(img_transpuesta.shape[0], img_transpuesta.shape[1])
            dst.write(mapa_clasificado_bloque.astype(np.uint8), window=window, indexes=1)
            print("¡Listo!")

print(f"\n¡PROCESO COMPLETADO! Carga el archivo '{IMAGEN_CLASIFICADA_SALIDA}' en QGIS.")

print("\n\n--- INFORME DE CLASIFICACIÓN (EVALUACIÓN DEL MODELO) ---")
print(informe_texto)
print("---------------------------------------------------------")