# BdE: Detección de Fraude en Transacciones

**Proyecto de ciencia de datos end-to-end** que implementa un sistema de detección de fraude utilizando datos públicos de pagos con tarjeta.  

El caso práctico se basa en transacciones minoristas (dataset público), pero la **metodología es transferible a sistemas de pagos de gran valor como TARGET2 o SNCE**, donde el Banco de España actúa como **operador y supervisor**.  

---

## 🚨 Contexto

El fraude y las anomalías en transacciones financieras suponen un **riesgo para la confianza y resiliencia de los sistemas de pago**.  
Este proyecto explora una solución que combina **modelos clásicos de ML** con **embeddings secuenciales** para capturar patrones ocultos en el comportamiento de los usuarios.  

---

## 📊 Dataset

- **Fuente:** FraudNLP (Boulieris et al., 2023)  
- **Volumen:** 105.302 transacciones (feb–oct 2020, banco europeo)  
- **Clases:**  
  - 99,9% legítimas (105.201)  
  - 0,096% fraudulentas (101)  
- **Contenido:**  
  - Secuencias completas de acciones de usuario en la web (≈1.900 tipos)  
  - Variables adicionales: importe, tiempos entre acciones, dispositivo, IP, frecuencia del beneficiario, etiqueta fraude/no fraude  

---

## 🧩 Metodología

### 1. Análisis Exploratorio (EDA)  
- Distribución de importes y acciones  
- Señales débiles de fraude → necesidad de ir más allá de datos tabulares  

### 2. Ingeniería de Features  
- Variables clásicas (importe, dispositivo, IP, etc.)  
- Estadísticos temporales (tiempo medio entre acciones, desviación típica, tiempo a la primera acción, total de acciones, etc.)  
- Embeddings secuenciales (128D) obtenidos con un **Transformer Encoder** que resume el historial de acciones de cada usuario  

### 3. Modelado  
- Algoritmo principal: **LightGBM**  
- Estrategias:  
  1. GB con features tabulares  
  2. GB + SMOTE  
  3. GB + Embeddings  
  4. GB + Embeddings + SMOTE  

- Métricas:  
  - **AU-PRC** (Average Precision-Recall Curve)  
  - **F1, F2, F0.5**  

## 4. Resultados

| Métrica   | GB    | GB + SMOTE | Encoder + GB | Encoder + GB + SMOTE |
|-----------|-------|------------|--------------|-----------------------|
| **AU-PRC** | 0.3300 | 0.2700     | 0.4400       | **0.5310**            |
| **F1**     | 0.3200 | 0.2940     | **0.5420**   | 0.4140                |
| **F2**     | 0.3210 | 0.2450     | **0.5600**   | 0.5510                |
| **F0.5**   | **0.5560** | 0.0000     | 0.5440       | 0.4810                |

**Conclusiones principales:**
- Los **embeddings secuenciales** elevan el rendimiento de forma notable (mejoras en todas las métricas frente al baseline).  
- El **uso aislado de SMOTE** degrada las métricas → solo es útil en combinación con embeddings.  
- Según la métrica priorizada:  
  - **Encoder + GB** es óptimo en F1 y F2 (mejor balance y recall).  
  - **Encoder + GB + SMOTE** alcanza el mejor AU-PRC.  

---

## ⚖️ Explicabilidad

- Los embeddings dificultan la interpretación directa  
- Se utilizan **SHAP values** para analizar la contribución de:  
  - Variables tabulares clásicas  
  - Dimensiones de embeddings secuenciales (`z_seq_x`)  

---

## ⚙️ Pipeline

- **Entrenamiento reproducible** con división estratificada (60/20/20)  
- **SMOTE** aplicado solo cuando es beneficioso  
- **Validación cruzada** con *Repeated Stratified K-Fold*  
- **Tracking y versionado** con **MLflow**  
- **CI/CD** con **GitHub Actions**  

---


