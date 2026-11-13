<div align="center">

# 🕵️ Fraud Detection Baseline

### Detección de fraude en tarjetas con ML: de un 0,17 % de casos positivos a un modelo que decide con criterio de negocio

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-NumPy-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/status-baseline%20completo-brightgreen)

</div>

---

## 🎯 El problema

En fraude con tarjetas, **casi todo es legítimo**. En el dataset de este proyecto solo el **0,17 %** de las transacciones son fraude, así que un modelo que siempre responda *"no es fraude"* acierta el **99,83 %** y no detecta ni un solo caso. Por eso la *accuracy* no sirve aquí.

Este mini proyecto construye un **baseline honesto** y mide lo que realmente importa: cuánto fraude se caza y cuánto dinero se ahorra.

## 📦 Datos

| | |
|---|---|
| **Dataset** | [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle / ULB) |
| **Transacciones** | 284.807 |
| **Variables** | `Time`, `Amount`, `V1…V28` (componentes PCA anonimizados) |
| **Objetivo** | `Class` (1 = fraude, 0 = legítima) |
| **Desbalanceo** | 99,83 % / 0,17 % |

> El CSV no se incluye en el repositorio. Descárgalo de Kaggle y colócalo como `creditcard.csv` en la raíz.

## 🧪 Metodología

```
creditcard.csv
      │
      ▼
 Split estratificado 60 / 20 / 20  ──►  train · validation · test
      │                                  (misma tasa de fraude en los tres)
      ▼
 Pipeline: StandardScaler → LogisticRegression(class_weight="balanced")
      │
      ▼
 Curva Precision-Recall + PR-AUC  (sobre validación)
      │
      ▼
 Umbral óptimo según coste de negocio  (100 € por fraude no detectado · 5 € por falsa alarma)
      │
      ▼
 Evaluación final, una sola vez, sobre test
```

**Decisiones clave**

- 🧱 **Validación y test separados.** El umbral se elige en validación y test se usa solo para el veredicto final, sin fuga de información.
- ⚖️ **`class_weight="balanced"`** para compensar el desbalanceo sin remuestrear.
- 📈 **PR-AUC en lugar de ROC-AUC ni accuracy**, porque con clases tan raras refleja mejor el rendimiento sobre la clase minoritaria.
- 💶 **Umbral por coste, no por defecto.** Un fraude que se escapa cuesta 20 veces más que una falsa alarma, así que 0,5 no es el punto de corte adecuado.

## 📊 Resultados

**PR-AUC en validación: `0,686`**, frente a `0,0017` de un clasificador aleatorio (≈ **400 veces mejor**).

### Test set (56.962 transacciones, 99 fraudes)

| Umbral | Fraudes cazados | Fraudes perdidos | Falsas alarmas | Precisión | Recall | F1 | Coste |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `0,50` (por defecto) | 90 | 9 | 1.310 | 0,06 | **0,91** | 0,12 | 7.450 € |
| **`0,98`** (óptimo por coste) | 85 | 14 | **71** | **0,54** | 0,86 | **0,67** | **1.755 €** |

### 💡 Conclusión

Subir el umbral de 0,5 a 0,98 supone **perder 5 fraudes más, pero evitar 1.239 falsas alarmas**. El coste total baja de **7.450 € a 1.755 €**, un **ahorro del 76 %** (calculado a partir de las matrices de confusión con los costes del enunciado).

En un caso real, cada falsa alarma es un cliente con la tarjeta bloqueada y un analista revisando. Con el umbral por defecto, el modelo tiene un 6 % de precisión: de cada 100 alertas, 94 son falsas.

## 🚀 Cómo ejecutarlo

```bash
git clone <url-del-repo>
cd fraud-baseline
pip install pandas numpy scikit-learn matplotlib jupyter
# coloca creditcard.csv en la raíz
jupyter notebook fraud_baseline.ipynb
```

## ⚠️ Limitaciones

- **Split aleatorio, no temporal.** En producción el modelo predice el futuro, así que habría que validar con un corte por `Time` para evitar optimismo.
- **El umbral óptimo (0,98) está casi en el borde de la rejilla** (0,01 a 0,99). Convendría refinar la búsqueda en el rango 0,95–0,999.
- **Costes fijados por supuesto** (100 € / 5 €). En un caso real, el coste de un fraude dependería de `Amount`.
- **Solo 99 fraudes en test**: las métricas tienen bastante varianza. Una validación cruzada estratificada daría intervalos más fiables.

## 🛣️ Próximos pasos

- [ ] Modelos de árboles (Random Forest, XGBoost / LightGBM) y comparación de PR-AUC
- [ ] Coste dependiente del importe de cada transacción (`Amount`)
- [ ] Validación temporal y validación cruzada estratificada
- [ ] Calibración de probabilidades
- [ ] Explicabilidad con SHAP
- [ ] Detección de anomalías no supervisada como señal complementaria

## 🗂️ Estructura

```
.
├── fraud_baseline.ipynb   # Análisis completo
├── creditcard.csv         # (no incluido) dataset de Kaggle
└── README.md
```

---

<div align="center">

Hecho por **Eugenio** · Matemáticas e IA (ICAI) · Interesado en fraude y ML aplicado

</div>
