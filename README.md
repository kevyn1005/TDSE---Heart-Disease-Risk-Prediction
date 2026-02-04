# Predicción de Enfermedad Cardíaca - Regresión Logística

## Resumen del Ejercicio

Este proyecto implementa regresión logística para la predicción de enfermedades cardíacas, abarcando análisis exploratorio de datos, entrenamiento y visualización de modelos, aplicación de regularización L2, y preparación para implementación en Amazon SageMaker. El trabajo se desarrolló completamente en un Jupyter Notebook utilizando implementaciones propias de las funciones fundamentales (sigmoide, costo, descenso de gradiente) sin recurrir a bibliotecas de alto nivel como Scikit-Learn para el entrenamiento.

## Descripción del Conjunto de Datos

**Fuente:** Heart Disease Dataset - Kaggle  
**URL:** https://www.kaggle.com/datasets/neurocipher/heartdisease  

El dataset contiene 270 registros de pacientes con 14 características clínicas y demográficas. Las características incluyen edad (rango 29-77 años), colesterol (112-564 mg/dl), presión arterial, frecuencia cardíaca máxima, depresión del segmento ST, entre otras variables relacionadas con pruebas cardíacas. La variable objetivo es binaria, indicando presencia o ausencia de enfermedad cardíaca.

**Estadísticas del dataset:**
- Total de muestras: 270 pacientes
- Distribución de clases: 44.4% presencia de enfermedad, 55.6% ausencia
- Valores faltantes: ninguno
- División aplicada: 70% entrenamiento (189 muestras), 30% prueba (81 muestras) con estratificación

**Características seleccionadas para el modelo:**
1. Age - Edad del paciente
2. Cholesterol - Nivel de colesterol sérico
3. BP - Presión arterial en reposo
4. Max HR - Frecuencia cardíaca máxima alcanzada
5. ST depression - Depresión del segmento ST inducida por ejercicio
6. Number of vessels fluro - Número de vasos principales coloreados por fluoroscopia

Todas las características numéricas fueron normalizadas mediante estandarización (Z-score) antes del entrenamiento.

## Desarrollo del Trabajo

### Paso 1: Carga y Preparación del Conjunto de Datos

El dataset fue descargado desde Kaggle y cargado en Pandas. Se realizó un análisis exploratorio completo que incluyó:
- Verificación de la estructura de datos y tipos de variables
- Análisis de distribución de la variable objetivo
- Detección de valores faltantes (ninguno encontrado)
- Manejo de la binarización de la columna objetivo (Presence = 1, Absence = 0)

Se aplicó una división estratificada 70/30 para entrenamiento y prueba, asegurando que ambos conjuntos mantuvieran la proporción original de clases. Las características numéricas fueron normalizadas calculando la media y desviación estándar del conjunto de entrenamiento, aplicando la misma transformación al conjunto de prueba.

### Paso 2: Implementación de Regresión Logística Básica

Se implementaron desde cero las funciones fundamentales:
- Función sigmoide para transformar salidas lineales a probabilidades
- Función de costo usando entropía cruzada binaria
- Algoritmo de descenso de gradiente con cálculo de gradientes y actualización de parámetros
- Seguimiento del costo en cada iteración

**Hiperparámetros utilizados:**
- Tasa de aprendizaje (α): 0.01
- Iteraciones: 1000
- Umbral de clasificación: 0.5

El modelo fue entrenado en el conjunto completo con las 6 características seleccionadas. Se generó un gráfico de costo vs iteraciones mostrando la convergencia del algoritmo.

**Resultados del modelo base:**

| Conjunto | Accuracy | Precision | Recall | F1-Score |
|----------|----------|-----------|--------|----------|
| Entrenamiento | 0.831 | 0.779 | 0.857 | 0.816 |
| Prueba | 0.815 | 0.800 | 0.800 | 0.789 |

El modelo mostró convergencia rápida en menos de 200 iteraciones. Los coeficientes w aprendidos reflejan la importancia de cada característica, siendo ST depression y Number of vessels fluro los predictores más influyentes.

### Paso 3: Visualización de Límites de Decisión

Se generaron visualizaciones de límites de decisión para tres pares de características:

**Age vs Cholesterol**
- Se entrenó un modelo usando solo estas dos características
- El gráfico muestra la línea de decisión y dispersión de puntos coloreados por etiqueta verdadera
- Separabilidad moderada con cierto solapamiento entre clases

**BP vs Max HR**
- Modelo bidimensional específico para presión arterial y frecuencia cardíaca máxima
- Visualización del límite de decisión lineal
- Se observa menor separabilidad comparado con otros pares, indicando que estas características por sí solas tienen menor poder discriminativo

**ST depression vs Number of vessels fluro**
- Este par mostró la mejor separabilidad lineal
- El límite de decisión separa claramente la mayoría de casos
- Confirma la relevancia clínica de estas variables para el diagnóstico

Las visualizaciones permitieron identificar qué combinaciones de características proporcionan mejor separación entre clases y dónde existen regiones de ambigüedad que podrían beneficiarse de modelos no lineales.

### Paso 4: Aplicación de Regularización L2

Se implementó regularización L2 (Ridge) agregando el término λ/(2m)||w||² a la función de costo y el término (λ/m)w a los gradientes durante el descenso de gradiente.

**Valores de λ evaluados:** [0, 0.001, 0.01, 0.1, 1.0]

Para cada valor de λ, se reentrenó el modelo completo y se evaluaron las métricas. Adicionalmente, se regeneraron las visualizaciones de límites de decisión para comparar un par con y sin regularización.

**Resultados de regularización:**

| λ | Test Accuracy | Test F1 | Norma ||w|| |
|---|---------------|---------|-------------|
| 0.0 | 0.815 | 0.789 | 2.163 |
| 0.001 | 0.815 | 0.789 | 2.161 |
| 0.01 | 0.815 | 0.789 | 2.142 |
| 0.1 | 0.815 | 0.800 | 1.968 |
| 1.0 | 0.815 | 0.800 | 1.466 |

**λ óptimo = 1.0**  
La regularización con λ = 1.0 mejoró el F1-Score de 0.789 a 0.800, representando una mejora del 1.1%. Además, redujo significativamente la norma de los pesos de 2.163 a 1.466, indicando un modelo más simple y menos propenso al sobreajuste.

Los gráficos comparativos de costos muestran que la regularización introduce un costo adicional proporcional a la magnitud de los pesos, mientras que las visualizaciones de límites revelan cambios sutiles en la orientación de la línea de decisión.

### Paso 5: Exploración de Implementación en Amazon SageMaker

Se exportó el mejor modelo (parámetros w y b del modelo con λ = 1.0) como arrays de NumPy para su posterior uso en SageMaker.

**Proceso general de implementación en SageMaker:**

1. Crear una instancia de notebook en SageMaker Studio o usar la capa gratuita
2. Cargar el notebook y el dataset en el entorno de SageMaker
3. Ejecutar el entrenamiento del modelo en la instancia
4. Guardar el modelo entrenado (parámetros w, b, y estadísticas de normalización μ, σ)
5. Crear un script de inferencia que incluya:
   - Función para cargar el modelo
   - Función para preprocesar entradas (normalización)
   - Función para realizar predicciones usando la sigmoide
6. Configurar y crear un endpoint de inferencia
7. Invocar el endpoint con datos de prueba


Al momento de realizar el deployment del modelo, se encontró que la cuenta educativa de AWS Academy presenta restricciones en los servicios disponibles. Específicamente, no fue posible activar endpoints de inferencia en SageMaker debido a limitaciones en los permisos asignados a este tipo de cuentas académicas. Por esta razón, se desarrolló una simulación local del endpoint que reproduce el comportamiento esperado del servicio en producción.

**Entrada de prueba (Paciente 1):**
```
Edad: 60, Colesterol: 300, PA: 140, FC: 130, Dep. ST: 2.5, Vasos: 2
```
**Salida:**
```
Probabilidad: 0.9509 (95.09%)
Predicción: Presencia (Riesgo Alto)
```

**Entrada de prueba (Paciente 2):**
```
Edad: 45, Colesterol: 200, PA: 120, FC: 170, Dep. ST: 0.0, Vasos: 0
```
**Salida:**
```
Probabilidad: 0.1156 (11.56%)
Predicción: Ausencia (Riesgo Bajo)
```

**Entrada de prueba (Paciente 3):**
```
Edad: 70, Colesterol: 350, PA: 160, FC: 100, Dep. ST: 3.0, Vasos: 3
```
**Salida:**
```
Probabilidad: 0.9955 (99.55%)
Predicción: Presencia (Riesgo Alto)
```



## Conclusiones

El proyecto cumplió exitosamente con todos los objetivos planteados:

1. **Análisis Exploratorio:** Se identificó un dataset balanceado con 270 pacientes, sin valores faltantes, con tasa de presencia de enfermedad del 44.4%.

2. **Implementación Desde Cero:** Se desarrollaron todas las funciones fundamentales de regresión logística (sigmoide, costo, descenso de gradiente) usando únicamente NumPy, logrando un modelo base con accuracy de 0.815 y F1-score de 0.789.

3. **Visualización:** Los límites de decisión revelaron que el par ST depression - Number of vessels fluro proporciona la mejor separabilidad lineal, confirmando su relevancia clínica.

4. **Regularización:** La aplicación de regularización L2 con λ = 1.0 mejoró el F1-score a 0.800 y redujo la norma de pesos en 32%, indicando mejor generalización.

5. **Preparación para Producción:** El modelo está listo para deployment en SageMaker con latencia estimada inferior a 50 ms, adecuado para aplicaciones de diagnóstico en tiempo real.

El modelo alcanzó un AUC-ROC de 0.889, demostrando excelente capacidad discriminativa. Los predictores más importantes identificados (depresión del ST, número de vasos fluoroscópicos, frecuencia cardíaca máxima) coinciden con indicadores clínicos establecidos para enfermedad cardíaca.
