# Quanto Tempo Dura um Processo?

Este projeto analisa o tempo de tramitação de processos judiciais no **TJPR (Tribunal de Justiça do Paraná)**, com foco em identificar:
- O tempo médio até a baixa (em dias);
- As variáveis que influenciam esse tempo (grau, tipo de procedimento, recursos, classe processual etc.);
- Perfis de processos por meio de **clusterização**.

---

## 📊 Metodologia
1. **Pré-processamento da base**  
   - Tratamento de valores nulos.  
   - Criação da variável `tempo_baixa` (diferença entre `Data de início` e `Data de referência`).  
   - Exclusão de colunas irrelevantes (dados de polo ativo/passivo, CNPJs etc.).  

2. **Modelos aplicados**  
   - **Regressão Linear (OLS):** identifica o impacto de variáveis no tempo até a baixa.  
   - **Random Forest + SHAP:** análise robusta de importância de variáveis e explicabilidade moderna.  
   - **Clusterização (KMeans):** identificação de grupos de processos com perfis semelhantes.  

---

## 📈 Resultados
- **Regressão Linear:** mostrou impacto forte dos tipos de procedimento (ex.: Execução Fiscal demora mais).  
- **Random Forest + SHAP:** reforçou os achados e mostrou a relevância da `Classe Processual`.  
- **Clusters:** dividiram os processos em 4 grupos distintos, de “rápidos” até “lentos”.  

---

## 📂 Estrutura do repositório

├── data/ # Bases originais e tratadas
├── notebooks/ # Códigos em Python (Spyder/Colab)
├── outputs/ # Resultados de análises e gráficos
├── README.md # Este arquivo
└── requirements.txt # Dependências do projeto
