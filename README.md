#  Steam Success Predictor

**Análisis del impacto del tipo de estudio (AAA, AA, Indie) en el rendimiento del mercado de Steam**

 *[Read this in English](README.en.md)*

Proyecto final del curso de IA del **Samsung Innovation Campus** — Grupo 2

**Autores:** Kevin Morales Cano · Adrián Contreras González

---

##  Descripción

Sistema analítico y predictivo basado en Machine Learning que estima el rendimiento comercial de videojuegos en Steam, clasificándolos en tres categorías de éxito a partir de variables de precio, plataforma, idiomas, fecha de lanzamiento y, especialmente, el historial de éxito del estudio desarrollador y de la editora.

El proyecto integra un pipeline completo de ingeniería de datos (limpieza, normalización de entidades, suavizado bayesiano) con el entrenamiento y comparación de varios algoritmos de clasificación supervisada.

##  Variable objetivo

La variable `exito_target` se define como una **clasificación multiclase** a partir de `Estimated owners`:

| Clase | Descripción | Umbral | % del dataset |
|---|---|---|---|
| 0 | Fracaso comercial | < 20.000 propietarios | ~80% |
| 1 | Éxito medio | 20.000 – 200.000 propietarios | ~16% |
| 2 | Gran éxito | > 200.000 propietarios | ~4% |

##  Estructura del repositorio

```
├── 01_data_cleaning_and_features.ipynb   # Limpieza, feature engineering y suavizado bayesiano
├── 02_model_training.ipynb               # Entrenamiento, evaluación y predicción de modelos
├── X_entrenamiento_limpio.csv            # Features de entrenamiento ya procesadas
├── y_entrenamiento_limpio.csv            # Target de entrenamiento
├── X_futuro_limpio.csv                   # Features de próximos lanzamientos
├── nombres_proximos_juegos.csv           # Nombres de los próximos lanzamientos (para trazabilidad)
├── modelo_lightgbm_steam_success.pkl     # Modelo final entrenado
├── scaler_steam_success.pkl              # StandardScaler ajustado sobre el train
└── README.md
```

>  **Sobre `games.csv`:** el dataset bruto original (~500 MB, obtenido de Kaggle) no está incluido en el repositorio por superar el límite de tamaño de archivo de GitHub. El notebook `01_data_cleaning_and_features.ipynb` requiere este archivo para ejecutarse desde cero — si necesitas reproducir el pipeline completo, solicítalo a los autores o descárgalo desde la fuente original en Kaggle. Para trabajar directamente con los modelos (notebook `02_model_training.ipynb`) no hace falta: los datasets ya procesados (`X_entrenamiento_limpio.csv`, `y_entrenamiento_limpio.csv`, `X_futuro_limpio.csv`, `nombres_proximos_juegos.csv`) sí están incluidos.

##  Pipeline de datos

```
[Datos brutos Steam] 
        ↓
[Limpieza textual y normalización de Developers/Publishers]
        ↓
[Ingeniería de características (suavizado Bayesiano)]
        ↓
[df_bueno (histórico)]     [df_proximos (futuros lanzamientos)]
        ↓                           ↓
        └───────────┬───────────────┘
                     ↓
      [Estandarización y escalado (StandardScaler)]
                     ↓
      [Modelado predictivo y validación]
```

**Puntos clave del preprocesamiento:**
- Reparación de un desalineamiento estructural en `games.csv` (cabecera corrupta que provocaba un desplazamiento de columnas desde `Price` en adelante).
- Normalización de nombres de desarrolladores/editoras mediante limpieza de texto, *fuzzy matching* (`SequenceMatcher`, umbral 0.82) y consolidación manual de marcas — de 72.859 a 70.723 entidades únicas.
- **Suavizado Bayesiano (Bayesian Target Encoding)** para `dev_exito_promedio` / `pub_exito_promedio`, evitando el sobreajuste en estudios con poco historial (`m = 5`).
- Extracción de ~1.000 próximos lanzamientos vía **Steam Web API**, con inyección manual de ~34 títulos de alta repercusión no indexados correctamente (p. ej. *Grand Theft Auto VI*).
- Prevención estricta de fuga de datos: el `scaler` se ajusta solo sobre el conjunto de entrenamiento, y ninguna estadística del histórico contamina el conjunto prospectivo.

##  Modelado

Se entrenaron y compararon cuatro algoritmos de clasificación multiclase:

| Modelo | Accuracy | F1 macro | ROC-AUC (ovr, macro) |
|---|---|---|---|
| **LightGBM**  | 0.911 | **0.838** | **0.981** |
| XGBoost | 0.906 | 0.832 | 0.979 |
| Random Forest | 0.903 | 0.825 | 0.976 |
| Regresión Logística | 0.865 | 0.755 | 0.940 |

**Modelo seleccionado: LightGBM**, por su mejor equilibrio entre las tres clases y su desempeño en la Clase 2 (Gran éxito: precisión 0.84, recall 0.74).

**Variables más influyentes:** `Price`, `dev_exito_promedio`, `año_lanzamiento`, `pub_exito_promedio`, `mes_lanzamiento` — resultado coherente con la hipótesis central del proyecto: el precio y el historial de éxito del estudio son los predictores dominantes del rendimiento comercial.

##  Limitaciones conocidas

- **Colisión de características en el conjunto prospectivo:** ~44% de los próximos lanzamientos comparten un vector de características idéntico (estudios sin historial que además coinciden en precio, plataformas y fecha), por lo que reciben la misma probabilidad predicha pese a ser juegos distintos.
- El modelo no incorpora señales de interés temprano (wishlists, seguidores) ni géneros específicos de forma granular — líneas de mejora futura.

##  Interfaz

El notebook `02_model_training.ipynb` incluye una interfaz interactiva ligera basada en `ipywidgets` para introducir las características de un juego hipotético y obtener la predicción de éxito en tiempo real, sin necesidad de modificar código.

##  Cómo ejecutar el proyecto

```bash
conda create -n steam-ml python=3.11 -y
conda activate steam-ml
conda install pandas numpy matplotlib seaborn scikit-learn jupyter ipywidgets -y
pip install xgboost lightgbm requests

jupyter notebook
```

1. Ejecuta `01_data_cleaning_and_features.ipynb` de principio a fin para regenerar los datasets limpios (o usa directamente los CSV ya incluidos en el repositorio).
2. Ejecuta `02_model_training.ipynb` para entrenar los modelos, ver las métricas y generar las predicciones.

>  **Nota sobre los CSV:** no abras ni resubas estos archivos a través de Google Drive/Sheets — puede corromper columnas numéricas de muchos decimales (convirtiéndolas en texto con formato de miles). Si necesitas compartirlos por Drive, hazlo dentro de un `.zip`.

##  Stack técnico

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `LightGBM` · `ipywidgets` · `Steam Web API`

##  Mejoras futuras

- Web scraping periódico para mantener el dataset actualizado en tiempo real.
- Interfaz web completa (Streamlit/Flask) más allá del notebook.
- Enriquecimiento de características para reducir la colisión de vectores idénticos.
- Optimización de hiperparámetros (`GridSearchCV`/`Optuna`) centrada en mejorar el recall de la Clase 2.

##  Autores

| | Rol |
|---|---|
| **Kevin Morales Cano** | Adquisición de datos, limpieza, normalización de entidades e ingeniería de características |
| **Adrián Contreras González** | Desarrollo de algoritmos, modelado, validación y prevención de fugas de datos |

Proyecto desarrollado en el marco del **Samsung Innovation Campus — Curso de IA**, cofinanciado por la Unión Europea, el Fondo Social Europeo Plus, la Escuela de Organización Industrial (EOI) y la Junta de Andalucía.
