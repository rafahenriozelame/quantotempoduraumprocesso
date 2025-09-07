# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path

# caminho do CSV
arquivo = Path(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado.csv")

# carregar (usando ; como separador, igual ao tjprsentenca)
df = pd.read_csv(arquivo, sep=";", low_memory=False)

# criar amostra de 1000 linhas aleatórias
df_sample = df.sample(n=1000, random_state=42)

# exportar para Excel
saida = Path(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado_amostra1000.xlsx")
df_sample.to_excel(saida, index=False)

print(f"✅ Amostra exportada com sucesso para: {saida}")

#%%

# caminho do CSV
arquivo = Path(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado.csv")

# carregar (usando ; como separador)
df = pd.read_csv(arquivo, sep=";", low_memory=False)

# mostrar informações gerais
print("Formato:", df.shape)  # número de linhas e colunas
print("\nColunas disponíveis:")
print(df.columns.tolist())

# mostrar as 5 primeiras linhas
print("\nAmostra (5 primeiras linhas):")
print(df.head(5))

#%% retirando colunas desnecessárias da base baixados


# caminho da base original
arquivo = Path(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado.csv")

# carregar CSV
df = pd.read_csv(arquivo, sep=";", low_memory=False)

# conferir quantidade de colunas
print("Total de colunas:", len(df.columns))

# selecionar colunas a remover (U até AD)
cols_to_drop = df.columns[20:30]   # U é a 21ª coluna, AD a 30ª

print("Colunas que serão removidas:", cols_to_drop.tolist())

# criar novo dataframe sem essas colunas
df_limpo = df.drop(columns=cols_to_drop)

# salvar em novo arquivo
saida = Path(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado_limpo.csv")
df_limpo.to_csv(saida, sep=";", index=False)

print(f"✅ Arquivo salvo em: {saida}")

#%%


df = pd.read_csv(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado.csv", sep=";", low_memory=False)

# Remover colunas U até AD
cols_to_drop = df.columns[20:29]  # ajustei para pegar até a 29ª
df_limpo = df.drop(columns=cols_to_drop)

# Salvar no mesmo diretório
saida = r"G:\My Drive\usp\TCC\base de dados\tjprbaixado_limpo.csv"
df_limpo.to_csv(saida, sep=";", index=False)

print(f"✅ Arquivo salvo em: {saida}")


#%%

# criar amostra de 50 linhas
amostra = df_limpo.sample(n=50, random_state=42)

# exportar para Excel
saida = r"G:\My Drive\usp\TCC\base de dados\tjprbaixado_limpo_amostra50.xlsx"
amostra.to_excel(saida, index=False)

print(f"✅ Amostra exportada em: {saida}")

#%% retira colunas de código classe e codigos assuntos, e relaciona a última classe com o nome da classe

# carregar a base original
df = pd.read_csv(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado.csv", sep=";", low_memory=False)

# remover colunas desnecessárias
df = df.drop(columns=['Codigos Classes', 'Codigos assuntos'])

# garantir que o código da última classe seja numérico
df['Codigo da Ultima classe'] = pd.to_numeric(df['Codigo da Ultima classe'], errors="coerce").astype("Int64")

# salvar em CSV (base completa, para uso em análise)
saida_csv = r"G:\My Drive\usp\TCC\base de dados\tjprbaixado_limpo.csv"
df.to_csv(saida_csv, sep=";", index=False)

print(f"✅ Base limpa salva em: {saida_csv}")

#%% fazendo o dicionário entre código ultima classe com o nome da última classe

# carregar a base já limpa
df = pd.read_csv(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado_limpo.csv", sep=";", low_memory=False)

# garantir que "Codigo da Ultima classe" é numérico
df['Codigo da Ultima classe'] = pd.to_numeric(df['Codigo da Ultima classe'], errors="coerce").astype("Int64")

# criar dicionário {codigo: nome}
mapa_classes = (
    df[['Codigo da Ultima classe', 'Nome da ultima classe']]
    .dropna()
    .drop_duplicates()
    .set_index('Codigo da Ultima classe')['Nome da ultima classe']
    .to_dict()
)

print("✅ Dicionário criado com sucesso!")
print("Exemplo (20 primeiros):")
print(dict(list(mapa_classes.items())[:20]))

# opcional: salvar em CSV/Excel para consulta
saida = r"G:\My Drive\usp\TCC\base de dados\mapa_classes.csv"
pd.DataFrame(list(mapa_classes.items()), columns=['Codigo da Ultima classe', 'Nome da ultima classe']).to_csv(saida, sep=";", index=False)

print(f"✅ Dicionário salvo em: {saida}")

#%% retirando a coluna com o nome da última classe

# carregar a base já limpa
df = pd.read_csv(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado_limpo.csv", sep=";", low_memory=False)

# remover a coluna "Nome da ultima classe"
if "Nome da ultima classe" in df.columns:
    df = df.drop(columns=["Nome da ultima classe"])

# salvar em novo CSV pronto para modelagem
saida = r"G:\My Drive\usp\TCC\base de dados\tjprbaixado_modelagem.csv"
df.to_csv(saida, sep=";", index=False)

print(f"✅ Base para modelagem salva em: {saida}")

#%% amostra 50 linhas

# carregar a base de modelagem
df = pd.read_csv(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado_modelagem.csv", sep=";", low_memory=False)

# gerar amostra de 50 linhas
amostra = df.sample(n=50, random_state=42)

# exportar para Excel para inspeção
saida = r"G:\My Drive\usp\TCC\base de dados\tjprbaixado_modelagem_amostra50.xlsx"
amostra.to_excel(saida, index=False)

print(f"✅ Amostra de 50 linhas salva em: {saida}")

#%% verificando a base:
    
# carregar a base de modelagem
df = pd.read_csv(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado_modelagem.csv", sep=";", low_memory=False)

# informações gerais
print("Formato:", df.shape)
print("\nColunas disponíveis:")
print(df.columns.tolist())

# amostra de 5 linhas
print("\nAmostra:")
print(df.head(5).T)  # .T para enxergar colunas como linhas

# tipos de dados
print("\nTipos de dados:")
print(df.dtypes)

# porcentagem de valores nulos por coluna
print("\nValores nulos (%):")
print(df.isna().mean().round(3) * 100)


#%% higienizando a base

# carregar a base atual
df = pd.read_csv(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado_modelagem.csv", sep=";", low_memory=False)

# colunas a remover
cols_to_drop = [
    'Tribunal',
    'Processo',
    'Codigo orgao',
    'id_municipio',
    'Polo ativo',
    'Polo ativo - CNPJ',
    'Polo ativo - Natureza juridica',
    'Polo ativo - CNAE',
    'Polo passivo',
    'Polo passivo - CNPJ',
    'Polo passivo - Natureza juridica',
    'Polo passivo - CNAE',
    'Poder publico'
]

# criar nova base apenas com colunas relevantes
df2 = df.drop(columns=cols_to_drop)

# salvar em novo CSV
saida = r"G:\My Drive\usp\TCC\base de dados\tjprbaixado_modelagem2.csv"
df2.to_csv(saida, sep=";", index=False)

print("Formato antes:", df.shape)
print("Formato depois:", df2.shape)
print(f"✅ Nova base salva em: {saida}")

#%% calculo tempo baixa e estatísticas descritivas

# carregar a base enxuta
df = pd.read_csv(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado_modelagem2.csv", sep=";", low_memory=False)

# converter colunas de data
df['Data de inicio'] = pd.to_datetime(df['Data de inicio'], errors="coerce", dayfirst=True)
df['Data de Referencia'] = pd.to_datetime(df['Data de Referencia'], errors="coerce", dayfirst=True)

# calcular tempo em dias
df['tempo_baixa'] = (df['Data de Referencia'] - df['Data de inicio']).dt.days

# estatísticas descritivas
print("\n📊 Estatísticas descritivas do tempo até a baixa:")
print(df['tempo_baixa'].describe(percentiles=[0.25, 0.5, 0.75]))

# salvar a nova base com a variável dependente
saida = r"G:\My Drive\usp\TCC\base de dados\tjprbaixado_modelagem3.csv"
df.to_csv(saida, sep=";", index=False)

print(f"\n✅ Base atualizada salva em: {saida}")

#%%

import matplotlib.pyplot as plt
import seaborn as sns

# carregar a base com tempo_baixa
df = pd.read_csv(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado_modelagem3.csv", sep=";", low_memory=False)

# garantir que tempo_baixa está numérico
df['tempo_baixa'] = pd.to_numeric(df['tempo_baixa'], errors='coerce')

# remover valores negativos ou absurdos (outliers extremos)
df = df[df['tempo_baixa'] >= 0]

# === Estatísticas descritivas ===
print("\n📊 Estatísticas do tempo até a baixa:")
print(df['tempo_baixa'].describe(percentiles=[0.25, 0.5, 0.75]))

# === Gráfico 1: Histograma ===
plt.figure(figsize=(10,6))
sns.histplot(df['tempo_baixa'], bins=100, kde=True, color="royalblue")
plt.title("Distribuição do tempo até a baixa (dias)", fontsize=14)
plt.xlabel("Tempo até a baixa (dias)")
plt.ylabel("Frequência")
plt.xlim(0, df['tempo_baixa'].quantile(0.99))  # zoom até p99 para evitar cauda longa
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# === Gráfico 2: Boxplot ===
plt.figure(figsize=(8,4))
sns.boxplot(x=df['tempo_baixa'], color="tomato")
plt.title("Boxplot do tempo até a baixa (dias)", fontsize=14)
plt.xlabel("Tempo até a baixa (dias)")
plt.xlim(0, df['tempo_baixa'].quantile(0.99))  # zoom até p99
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# === Gráfico 3: Boxplot por Grau ===
plt.figure(figsize=(10,6))
sns.boxplot(x="Grau", y="tempo_baixa", data=df, palette="Set2")
plt.title("Tempo até a baixa por Grau de Jurisdição", fontsize=14)
plt.ylabel("Tempo até a baixa (dias)")
plt.ylim(0, df['tempo_baixa'].quantile(0.99))  # limitar p99
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# === Gráfico 4: Média por Grau ===
plt.figure(figsize=(8,5))
df.groupby("Grau")['tempo_baixa'].mean().sort_values().plot(kind="bar", color="seagreen")
plt.title("Tempo médio até a baixa por Grau de Jurisdição", fontsize=14)
plt.ylabel("Tempo médio (dias)")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

#%% ajuste para regressão linear

import pandas as pd
import statsmodels.api as sm
import numpy as np

# carregar base
df = pd.read_csv(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado_modelagem3.csv", sep=";", low_memory=False)

# garantir variável dependente correta
df['tempo_baixa'] = pd.to_numeric(df['tempo_baixa'], errors='coerce')
df = df[df['tempo_baixa'] >= 0]

# selecionar variáveis explicativas
X = df[['Grau','UF','Municipio','Ano','Mes',
        'Codigo da Ultima classe','Formato','Procedimento',
        'Recurso Originario','Recurso']]

# dummizar categóricas
X = pd.get_dummies(X, drop_first=True)

# converter tudo para float
X = X.astype(float)

# variável dependente (log-transformada para suavizar cauda longa)
y = np.log1p(df['tempo_baixa'])

# adicionar constante (intercepto)
X = sm.add_constant(X)

# rodar regressão
modelo = sm.OLS(y, X).fit()

print(modelo.summary())


#%% regressão sem variável município

import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# carregar base
df = pd.read_csv(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado_modelagem3.csv", sep=";", low_memory=False)

# variável dependente
df['tempo_baixa'] = pd.to_numeric(df['tempo_baixa'], errors='coerce')
df = df[df['tempo_baixa'] >= 0]

# selecionar variáveis explicativas (sem Município e sem Nome orgao)
X = df[['Grau','UF','Ano','Mes',
        'Codigo da Ultima classe','Formato','Procedimento',
        'Recurso Originario','Recurso']]

# transformar categóricas em dummies
X = pd.get_dummies(X, drop_first=True)

# converter para float
X = X.astype(float)

# log-transformar Y
y = np.log1p(df['tempo_baixa'])

# adicionar intercepto
X = sm.add_constant(X)

# rodar regressão
modelo = sm.OLS(y, X).fit()

print(modelo.summary())

# === Gráfico dos coeficientes ===
coefs = modelo.params.sort_values()
coefs_to_plot = pd.concat([coefs.head(15), coefs.tail(15)])

plt.figure(figsize=(8,8))
coefs_to_plot.plot(kind="barh", color=["tomato" if v<0 else "seagreen" for v in coefs_to_plot])
plt.title("Coeficientes da Regressão Linear (sem Municípios)")
plt.xlabel("Impacto no log(tempo_baixa)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

#%% resumo coeficientes

import matplotlib.pyplot as plt
import pandas as pd

# pegar coeficientes e ordenar
coefs = modelo.params.sort_values()

# filtrar apenas variáveis principais (ignorar constante e dummies raros)
coefs_to_plot = coefs.drop("const")

plt.figure(figsize=(8,6))
coefs_to_plot.plot(kind="barh", color=["tomato" if v<0 else "seagreen" for v in coefs_to_plot])
plt.title("Impacto estimado das variáveis no log(tempo até a baixa)")
plt.xlabel("Coeficiente da regressão linear")
plt.axvline(0, color="black", linewidth=0.8)
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

#%%

import pandas as pd

# carregar base
df = pd.read_csv(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado_modelagem3.csv", sep=";", low_memory=False)

# garantir que tempo_baixa é numérico
df['tempo_baixa'] = pd.to_numeric(df['tempo_baixa'], errors='coerce')

# contar registros problemáticos
nulos = df['tempo_baixa'].isna().sum()
total = len(df)
negativos = (df['tempo_baixa'] < 0).sum()

print("📊 Diagnóstico da variável tempo_baixa")
print(f"Total de linhas: {total:,}")
print(f"Com NaN (faltando): {nulos:,} ({(nulos/total)*100:.2f}%)")
print(f"Com valores negativos: {negativos:,} ({(negativos/total)*100:.2f}%)")
print(f"Válidos: {total - nulos - negativos:,} ({((total-nulos-negativos)/total)*100:.2f}%)")

#%%

import pandas as pd

# carregar base original baixada
df = pd.read_csv(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado.csv", sep=";", low_memory=False)

# tentar converter as colunas de datas
for col in ["Data de inicio", "Data de Referencia", 
            "Data do julgamento: Situacao: Movimento", 
            "Data da decisao: Situacao: Movimento"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        validos = df[col].notna().sum()
        print(f"{col}: {validos:,} datas válidas ({(validos/len(df))*100:.2f}%)")
        
        
#%%


import pandas as pd

# carregar base original
df_orig = pd.read_csv(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado.csv", sep=";", low_memory=False)

print("Formato:", df_orig.shape)
print("\nColunas disponíveis:")
print(df_orig.columns.tolist())

# checar todas as colunas que têm "Data" no nome
for col in df_orig.columns:
    if "Data" in col:
        df_orig[col] = pd.to_datetime(df_orig[col], errors="coerce", dayfirst=True)
        validos = df_orig[col].notna().sum()
        print(f"{col}: {validos:,} datas válidas ({(validos/len(df_orig))*100:.2f}%)")
        
#%% frequencia das linhas que não tem as datas

import pandas as pd

# carregar base original
df_orig = pd.read_csv(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado.csv", sep=";", low_memory=False)

# converter datas
df_orig['Data de inicio'] = pd.to_datetime(df_orig['Data de inicio'], errors="coerce", dayfirst=True)
df_orig['Data de Referencia'] = pd.to_datetime(df_orig['Data de Referencia'], errors="coerce", dayfirst=True)

# filtrar casos sem nenhuma das duas datas
sem_ambas = df_orig[df_orig['Data de inicio'].isna() & df_orig['Data de Referencia'].isna()]

# tabela de frequências por classe
freq_classes = sem_ambas['Nome da ultima classe'].value_counts().reset_index()
freq_classes.columns = ['Classe', 'Qtd_processos']

# salvar para análise no Excel
saida = r"G:\My Drive\usp\TCC\base de dados\freq_classes_sem_datas.xlsx"
freq_classes.to_excel(saida, index=False)

print("Total de processos sem datas:", sem_ambas.shape[0])
print("✅ Frequências por classe salvas em:", saida)
print("\nTop 10 classes sem datas:")
print(freq_classes.head(10))

#%%

import pandas as pd

# carregar base original
df_orig = pd.read_csv(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado.csv", sep=";", low_memory=False)

# garantir que Data de Referencia está como datetime
df_orig['Data de Referencia'] = pd.to_datetime(df_orig['Data de Referencia'], errors="coerce", dayfirst=True)

# filtrar processos sem data de baixa
sem_baixa = df_orig[df_orig['Data de Referencia'].isna()]

# pegar 10 exemplos
amostra10 = sem_baixa.sample(10, random_state=42)

# exportar para Excel
saida = r"G:\My Drive\usp\TCC\base de dados\amostra10_sem_baixa.xlsx"
amostra10.to_excel(saida, index=False)

print("✅ Amostra salva em:", saida)
print(amostra10[['Processo','Nome da ultima classe','Procedimento','Grau']])


#%%

import pandas as pd

df_orig = pd.read_csv(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado.csv", sep=";", low_memory=False)

# converter para datas
df_orig['Data de inicio'] = pd.to_datetime(df_orig['Data de inicio'], errors="coerce", dayfirst=True)
df_orig['Data de Referencia'] = pd.to_datetime(df_orig['Data de Referencia'], errors="coerce", dayfirst=True)

# contar casos completos
completos = df_orig[df_orig['Data de inicio'].notna() & df_orig['Data de Referencia'].notna()]
print("📊 Processos com ambas as datas preenchidas:", completos.shape[0])
print("Proporção sobre o total:", completos.shape[0] / len(df_orig) * 100, "%")

#%% base real

# criar base real
df_real = df_orig[df_orig['Data de inicio'].notna() & df_orig['Data de Referencia'].notna()].copy()

# calcular tempo
df_real['tempo_baixa'] = (df_real['Data de Referencia'] - df_real['Data de inicio']).dt.days
df_real = df_real[df_real['tempo_baixa'] >= 0]  # descartar inconsistentes

print("Base real final:", df_real.shape)

# salvar para uso nos modelos
saida = r"G:\My Drive\usp\TCC\base de dados\tjprbaixado_base_real.csv"
df_real.to_csv(saida, sep=";", index=False)
print(f"✅ Base real salva em: {saida}")

#%% regressão base real

import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# carregar base real
df = pd.read_csv(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado_base_real.csv", sep=";", low_memory=False)

# carregar dicionário de classes (Código ↔ Nome)
mapa_classes = pd.read_csv(r"G:\My Drive\usp\TCC\base de dados\mapa_classes.csv", sep=";")
mapa_dict = dict(zip(mapa_classes['Codigo da Ultima classe'], mapa_classes['Nome da ultima classe']))

# preparar variáveis
df['tempo_baixa'] = pd.to_numeric(df['tempo_baixa'], errors='coerce')
df = df[df['tempo_baixa'] >= 0]

X = df[['Grau','UF','Ano','Mes',
        'Codigo da Ultima classe','Formato','Procedimento',
        'Recurso Originario','Recurso']]
X = pd.get_dummies(X, drop_first=True)
X = X.astype(float)

y = np.log1p(df['tempo_baixa'])
X = sm.add_constant(X)

# rodar regressão
modelo = sm.OLS(y, X).fit()
print(modelo.summary())

# gráfico de coeficientes
coefs = modelo.params.sort_values()
coefs_to_plot = coefs.drop("const")

plt.figure(figsize=(8,8))
coefs_to_plot.plot(kind="barh", color=["tomato" if v<0 else "seagreen" for v in coefs_to_plot])
plt.title("Impacto das variáveis no log(tempo até a baixa)")
plt.xlabel("Coeficiente da regressão linear")
plt.axvline(0, color="black", linewidth=0.8)
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

#%% random 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import shap

# preparar X e y
X = df[['Grau','UF','Ano','Mes',
        'Codigo da Ultima classe','Formato','Procedimento',
        'Recurso Originario','Recurso']]
X = pd.get_dummies(X, drop_first=True)
y = df['tempo_baixa']

# split treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# treinar modelo
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# avaliar
pred = rf.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
print("R²:", r2_score(y_test, pred))

# importância das variáveis
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)

#%% cluster

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# selecionar variáveis
X = df[['tempo_baixa','Ano','Mes','Codigo da Ultima classe']]
X = pd.get_dummies(df[['Grau','Formato','Procedimento']], drop_first=True).join(X)

# padronizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# rodar KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# gráfico 1: distribuição dos clusters
plt.figure(figsize=(7,5))
df['cluster'].value_counts().sort_index().plot(kind="bar", color="seagreen")
plt.title("Distribuição dos processos por cluster")
plt.xlabel("Cluster")
plt.ylabel("Número de processos")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# gráfico 2: tempo por cluster
plt.figure(figsize=(9,6))
sns.boxplot(x="cluster", y="tempo_baixa", data=df, palette="Set2")
plt.title("Tempo até a baixa por cluster")
plt.ylabel("Tempo (dias)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# salvar versão com clusters
saida = r"G:\My Drive\usp\TCC\base de dados\tjprbaixado_clusters.csv"
df.to_csv(saida, sep=";", index=False)
print(f"✅ Base com clusters salva em: {saida}")


plt.figure(figsize=(8,6))
importances.tail(20).plot(kind="barh", color="royalblue")
plt.title("Top 20 variáveis mais importantes - Random Forest")
plt.xlabel("Importância relativa")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# SHAP Values
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test[:2000])

shap.summary_plot(shap_values, X_test[:2000], plot_type="bar")   # impacto global
shap.summary_plot(shap_values, X_test[:2000])                   # distribuição por variável

#%%

import pandas as pd

# carregar base com clusters já gerados
df = pd.read_csv(r"G:\My Drive\usp\TCC\base de dados\tjprbaixado_clusters.csv", sep=";", low_memory=False)

# carregar dicionário de classes
mapa_classes = pd.read_csv(r"G:\My Drive\usp\TCC\base de dados\mapa_classes.csv", sep=";")
mapa_dict = dict(zip(mapa_classes['Codigo da Ultima classe'], mapa_classes['Nome da ultima classe']))

# traduzir código da última classe
df['Nome da Classe'] = df['Codigo da Ultima classe'].map(mapa_dict)

# === 1. Estatísticas por cluster ===
resumo_clusters = df.groupby("cluster")['tempo_baixa'].agg(
    n="count",
    media="mean",
    mediana="median",
    minimo="min",
    maximo="max"
).reset_index()

print("📊 Estatísticas por cluster:")
print(resumo_clusters)

# salvar em Excel
resumo_clusters.to_excel(r"G:\My Drive\usp\TCC\base de dados\resumo_clusters.xlsx", index=False)

# === 2. Top matérias por cluster ===
top_classes_cluster = (
    df.groupby(["cluster","Nome da Classe"])
      .size()
      .reset_index(name="Qtd")
)

# pegar top 5 matérias de cada cluster
top5_por_cluster = (
    top_classes_cluster
    .sort_values(["cluster","Qtd"], ascending=[True,False])
    .groupby("cluster")
    .head(5)
)

print("\n📌 Top 5 matérias por cluster:")
print(top5_por_cluster)

# salvar em Excel
top5_por_cluster.to_excel(r"G:\My Drive\usp\TCC\base de dados\top5_classes_por_cluster.xlsx", index=False)

print("\n✅ Arquivos salvos:")
print(" - resumo_clusters.xlsx")
print(" - top5_classes_por_cluster.xlsx")


#%%

import matplotlib.pyplot as plt
import seaborn as sns

# usar a tabela já criada com top 5 matérias
top5 = top5_por_cluster.copy()

# configurar estilo
sns.set(style="whitegrid")

# loop para cada cluster
for c in sorted(top5['cluster'].unique()):
    dados = top5[top5['cluster'] == c].sort_values("Qtd", ascending=True)

    plt.figure(figsize=(10,5))
    sns.barplot(
        x="Qtd", y="Nome da Classe",
        data=dados, palette="Set2"
    )
    plt.title(f"Top 5 matérias no Cluster {c}", fontsize=14, weight="bold")
    plt.xlabel("Quantidade de processos")
    plt.ylabel("Matéria (Nome da Classe)")
    plt.tight_layout()

    # salvar gráfico em PNG
    plt.savefig(rf"G:\My Drive\usp\TCC\base de dados\cluster_{c}_top5.png")
    plt.show()

#%%

import matplotlib.pyplot as plt
import seaborn as sns

# calcular tempo médio por cluster e matéria
tempo_medio = (
    df.groupby(["cluster", "Nome da Classe"])['tempo_baixa']
      .mean()
      .reset_index()
)

# pegar só as matérias mais frequentes (top 5 por cluster)
top5_materias = (
    df.groupby(["cluster","Nome da Classe"])
      .size()
      .reset_index(name="Qtd")
      .sort_values(["cluster","Qtd"], ascending=[True,False])
      .groupby("cluster")
      .head(5)
)

# filtrar apenas essas matérias
tempo_medio_top5 = tempo_medio.merge(top5_materias[["cluster","Nome da Classe"]],
                                     on=["cluster","Nome da Classe"],
                                     how="inner")

# gráfico
plt.figure(figsize=(14,7))
sns.barplot(
    x="tempo_baixa", y="Nome da Classe",
    hue="cluster", data=tempo_medio_top5,
    palette="Set2"
)

plt.title("Tempo médio de tramitação (em dias) por matéria e cluster", fontsize=14, weight="bold")
plt.xlabel("Tempo médio (dias)")
plt.ylabel("Matéria")
plt.legend(title="Cluster")
plt.tight_layout()

# salvar gráfico
plt.savefig(r"G:\My Drive\usp\TCC\base de dados\tempo_medio_materia_cluster.png")
plt.show()

#%%


plt.figure(figsize=(16,7))
ax = sns.barplot(
    x="Nome da Classe", y="tempo_baixa",
    hue="cluster", data=tempo_medio_top5,
    palette="Set2"
)

# títulos
plt.title("Tempo médio de tramitação (em dias) por matéria e cluster", fontsize=14, weight="bold")
plt.ylabel("Tempo médio (dias)")
plt.xlabel("Matéria (Nome da Classe)")
plt.xticks(rotation=45, ha="right")

# adicionar rótulos em cima das barras
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.0f}', 
                (p.get_x() + p.get_width()/2., height),
                ha='center', va='bottom', fontsize=9, color="black", rotation=0)

plt.tight_layout()
plt.savefig(r"G:\My Drive\usp\TCC\base de dados\tempo_medio_materia_cluster_invertido.png")
plt.show()






