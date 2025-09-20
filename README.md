# 🚢 Titanic Dashboard — Portada del proyecto

**Primer proyecto del bootcamp _CSIC Data&IA_ — Upgrade-Hub**
Este repositorio contiene un dashboard interactivo desarrollado en **Streamlit** para realizar un Análisis Exploratorio de Datos (EDA) sobre el dataset del Titanic. Este trabajo corresponde al primer desafío práctico del bootcamp *CSIC Data&IA* de Upgrade‑Hub y fue diseñado para ofrecer una experiencia clara, reproducible y enfocada en la visualización de hallazgos.

---
## 🎯 Resumen ejecutivo (qué hicimos)

Creamos una aplicación web ligera que permite cargar y filtrar el dataset del Titanic, explorar estadísticas descriptivas, visualizar relaciones clave entre variables y generar gráficos personalizados. El objetivo fue transformar datos crudos en insights accionables para un público técnico y no técnico.
Contrato del proyecto (inputs / outputs / criterios de éxito):

- Inputs: `data/titanic_clean.csv` (por defecto) o CSV personalizado subido por el usuario.
- Outputs: panel interactivo con visualizaciones, métricas clave y descarga del subconjunto filtrado.
- Criterios de éxito: la app debe permitir filtrar datos, generar las visualizaciones principales y exportar los datos filtrados sin errores evidentes.

---
## 🧭 Estructura del dashboard (secciones en `app.py`)

La aplicación está organizada en cuatro pestañas principales, que reflejan las acciones más relevantes en un flujo de EDA:
1. 📊 Data

- Vista previa de los datos filtrados.
- Controles para mostrar información del dataset (dimensiones, tipos).
- Descarga del subconjunto filtrado en formato CSV.

2. 🔍 EDA (Exploratory Data Analysis)
- Estadísticas descriptivas (describe).
- Visualizaciones predeterminadas: histograma de edad, supervivencia por sexo y por clase, boxplots de tarifa por clase y por puerto, y mapa de correlación entre variables numéricas.

3. 📈 Metrics
- Métricas en tiempo real: total de pasajeros, tasa de supervivencia, edad promedio y tarifa promedio.
- Distribuciones por sexo y por clase con porcentajes.

4. 🎨 Custom Plots
- Herramienta para crear gráficos personalizados (histograma, conteo, boxplot y scatter).
- Selección dinámica de variables X/Y y opción de agrupar/colorear por una variable con pocas categorías.

---
## ✨ Funcionalidades destacadas

- Carga alternativa de datos: uso del CSV por defecto o carga manual desde la interfaz.
- Filtros interactivos en la barra lateral: sexo, clase, puerto de embarque, supervivencia, rango de edad y rango de tarifa.
- Visualizaciones que responden inmediatamente a los filtros aplicados.
- Descarga de los datos filtrados para análisis posteriores.
- Diseño defensivo: manejo de errores en carga y generación de gráficos; fallback a dataset sintético si no hay CSV disponible.

---
## 🛠 Tecnologías y dependencias

- Lenguaje: Python
- Framework UI: Streamlit
- Manipulación y análisis: pandas, numpy
- Visualizaciones: Plotly Express (y plotly)
- Otras utilidades: seaborn, scikit-learn (mencionadas en instalación)

Dependencias recomendadas (puedes instalarlas con pip):
```powershell
pip install streamlit pandas numpy plotly seaborn scikit-learn
```

---
## ▶️ Cómo ejecutar la aplicación

1. Clona este repositorio o descarga los archivos al directorio local.
2. Asegúrate de tener Python 3.8+ y las dependencias instaladas.
3. Ejecuta:

```powershell
streamlit run app.py
```

La app abrirá una pestaña en tu navegador (por defecto http://localhost:8501) donde podrás interactuar con el dashboard.
---

## ✅ Buenas prácticas y notas técnicas (como Project Manager)

- Mantuvimos separación clara entre lógica de carga, filtrado y generación de gráficos.
- El código utiliza caching (`@st.cache_data`) para optimizar la recarga del dataset.
- Recomendación: añadir un fichero `requirements.txt` para fijar versiones antes de compartir o desplegar.

Posibles mejoras a corto plazo:
- Agregar pruebas unitarias básicas para las funciones de filtrado y generación de gráficos.
- Añadir validación más estricta del schema del CSV subido.
- Implementar historias de usuario y un changelog en el repositorio.

---
## 📁 Estructura del repositorio

- `app.py` — Código fuente principal (dashboard Streamlit).
- `data/titanic_clean.csv` — Dataset limpio (esperado por defecto).
- `README.md` — Este documento.

---
## 🧾 Créditos y contexto

Este trabajo es el primer proyecto práctico del bootcamp **CSIC Data&IA** impartido por **Upgrade‑Hub**. Fue desarrollado con fines educativos para practicar EDA, visualización y buenas prácticas de ingeniería reproducible.

---


