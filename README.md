# 🗳️ Análise de Despesas de Candidatos - Eleições 2024 (Databricks)

Este projeto consiste em um notebook Python desenvolvido para a plataforma **Databricks**, focado na ingestão, limpeza, análise exploratória e clusterização de dados de despesas de candidatos nas Eleições Municipais de 2024.

O script combina a capacidade de processamento distribuído do **PySpark** para ingestão com a flexibilidade do **Pandas** e **Scikit-learn** para análises refinadas e Machine Learning.

## 📋 Funcionalidades do Projeto

O pipeline executa as seguintes etapas automaticamente:

1.  **Ingestão Híbrida:** Carrega dados de uma tabela Delta/Hive via Spark e converte para Pandas para manipulação granular.
2.  **Detecção Inteligente de Colunas:** Identifica automaticamente colunas críticas (Valor, UF, Partido, Cargo, Candidato) independente de pequenas variações nos nomes.
3.  **Limpeza de Dados:**
    * Conversão de tipos de dados.
    * Remoção de valores nulos ou zerados.
    * Tratamento de outliers para visualização.
4.  **Visualização de Dados (Data Viz):** Geração automática de gráficos estatísticos (Histogramas, Barras, Pizza e Boxplots).
5.  **Machine Learning (Clustering):** Aplicação do algoritmo **K-Means** para agrupar candidatos com perfis de gastos semelhantes baseados no volume total e quantidade de despesas.

## 🛠 Ferramentas

* **Plataforma:** Databricks (Runtime ML recomendado)
* **Linguagem:** Python 3.x
* **Bibliotecas Principais:**
    * `pyspark`: Leitura da tabela fonte.
    * `pandas` & `numpy`: Manipulação e limpeza dos dados.
    * `matplotlib` & `seaborn`: Visualização de dados.
    * `scikit-learn`: Pré-processamento (StandardScaler) e Clusterização (KMeans).

## 📊 Visualizações Geradas

O notebook gera automaticamente os seguintes insights visuais:

* **Distribuição de Valores:** Histograma com marcação de média e mediana (filtrado pelo percentil 95).
* **Geografia do Gasto:** Top estados com maior volume de despesas.
* **Partidos:** Ranking dos 10 partidos com maiores gastos.
* **Cargos:** Distribuição percentual (Pizza) e variação de valores por cargo (Boxplot).
* **Clusters de Candidatos:** Gráfico de dispersão mostrando os grupos identificados pelo algoritmo K-Means.

## 🚀 Como Executar

### Pré-requisitos

1.  Acesso a um workspace Databricks.
2.  Tabela de dados carregada no catálogo com o nome: `workspace.default.despesas_candidatos_2024_` (ou ajuste a variável `TABELA` no Passo 2 do notebook).

### Passo a Passo

1.  Importe o arquivo `.dbc` ou `.py` para o seu Workspace.
2.  Certifique-se de que o cluster está ativo.
3.  Verifique se o nome da tabela na célula do **Passo 2** corresponde à sua tabela real.
4.  Execute todas as células (`Run All`).

## 🧠 Detalhes da Modelagem (K-Means)

Para a clusterização dos candidatos, foram utilizadas as seguintes features:

* `total_despesas`: Soma do valor declarado.
* `num_despesas`: Quantidade de lançamentos.

Os dados foram normalizados utilizando `StandardScaler` antes da aplicação do algoritmo K-Means, configurado para identificar **4 perfis de comportamento** (clusters).
