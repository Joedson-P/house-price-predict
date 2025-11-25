# Previsão de Preços de Casas (House Price Prediction)

Este projeto de **Regressão** foca na previsão do preço de venda (`SalePrice`) de casas na região de Ames, Iowa, utilizando um conjunto de dados complexo de 80 features.

O objetivo é construir um modelo preditivo altamente preciso para precificação imobiliária.

---

## Metodologia

1.  **Análise Exploratória (EDA):** Identificação de *outliers* e **assimetria positiva** no `SalePrice`.
2.  **Pré-processamento e Feature Engineering:**
    * Tratamento de *Missing Data* (distinguindo 'None' para ausência de recurso e imputação estatística para dados perdidos).
    * Transformação do *target* (`SalePrice`) via **$\log(1+x)$** para normalizar a distribuição.
    * Criação de *features* combinadas cruciais (ex: `TotalSF` e `Age`).
    * Codificação categórica via **One-Hot Encoding** (gerando 334 features).
3.  **Modelagem e Otimização:** Uso de Regressão de Ridge como *baseline* e otimização do **XGBoost Regressor** via Grid Search focado na minimização do **RMSE**.

---

## Tecnologias e Dados

* **Fonte de Dados:** [House Prices - Advanced Regression Techniques (Kaggle)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
* **Linguagem:** Python
* **Bibliotecas:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib/Seaborn.

---

## Resultados Finais e Insights

O **XGBoost Regressor Otimizado** foi o modelo selecionado, alcançando uma redução de mais de 40% no erro de previsão em relação ao *baseline* de Ridge.

|1|2|3|
| :--- | :--- | :--- |
| **Modelo Selecionado** | XGBoost Regressor Otimizado | Configurado com `lr=0.05`, `max_depth=3`, `n_estimators=300`. |
| **RMSE (Erro Médio)** | **$14,046.04$** | Erro médio de previsão na escala original (em dólares). |

### Principais Fatores Predições de Preço

O modelo priorizou as seguintes *features* para determinar o preço:

1.  **OverallQual (Qualidade Geral):** Fator mais importante, refletindo a qualidade do material e acabamento.
2.  **TotalSF (Área Total):** O tamanho total da área habitável (Baseado nas suas *features* engenheiradas).
3.  **CentralAir\_N (Ausência de Ar Central):** Um forte preditor negativo.

---

## Entrega

O modelo otimizado foi serializado e salvo para *deploy*: `models/xgb_regressor_optimized.pkl`.
