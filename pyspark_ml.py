# Databricks notebook source
# MAGIC %md
# MAGIC ## 📚 
# MAGIC

# COMMAND ----------

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configurar para evitar erro do threadpoolctl
import os
os.environ['OMP_NUM_THREADS'] = '1'

print("=" * 80)
print("BIBLIOTECAS IMPORTADAS")
print("=" * 80)
print(f"✅ Pandas: {pd.__version__}")
print(f"✅ NumPy: {np.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📥 Passo 2: Carregar Dados

# COMMAND ----------

# Nome da tabela
TABELA = "workspace.default.despesas_candidatos_2024_"

print("=" * 80)
print("CARREGANDO DADOS")
print("=" * 80)
print(f"📊 Tabela: {TABELA}")

# Carregar com Spark
df_spark = spark.table(TABELA)

# Converter para Pandas
df = df_spark.toPandas()

print(f"\n✅ DADOS CARREGADOS!")
print(f"   • Registros: {len(df):,}")
print(f"   • Colunas: {len(df.columns)}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔍 Passo 3: Explorar Dados

# COMMAND ----------

print("\n📋 COLUNAS DISPONÍVEIS:")
print("=" * 80)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")
print("=" * 80)

# COMMAND ----------

# Mostrar primeiras linhas
print("\n📋 PRIMEIRAS 5 LINHAS:")
display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔄 Passo 4: Detectar e Processar Colunas

# COMMAND ----------

print("\n🔍 DETECTANDO COLUNAS...")
print("=" * 80)

# Mapear colunas
mapeamento = {}

# Coluna de valor
for col in df.columns:
    if any(x in col.upper() for x in ['VALOR', 'DESPESA', 'VR_']):
        mapeamento['valor'] = col
        print(f"✅ VALOR: {col}")
        break

# Coluna de UF
for col in df.columns:
    if any(x in col.upper() for x in ['UF', 'ESTADO', 'SG_UF']):
        mapeamento['uf'] = col
        print(f"✅ UF: {col}")
        break

# Coluna de partido
for col in df.columns:
    if 'PARTIDO' in col.upper():
        mapeamento['partido'] = col
        print(f"✅ PARTIDO: {col}")
        break

# Coluna de cargo
for col in df.columns:
    if 'CARGO' in col.upper():
        mapeamento['cargo'] = col
        print(f"✅ CARGO: {col}")
        break

# Coluna de candidato
for col in df.columns:
    if any(x in col.upper() for x in ['CANDIDATO', 'NOME']):
        mapeamento['candidato'] = col
        print(f"✅ CANDIDATO: {col}")
        break

print(f"\n✅ {len(mapeamento)} colunas detectadas")
print("=" * 80)

# COMMAND ----------

# Verificar se encontrou coluna de valor
if 'valor' not in mapeamento:
    print("\n❌ ERRO: Coluna de valores não encontrada!")
    print("\nColunas disponíveis:")
    for col in df.columns:
        print(f"  - {col}")
    raise ValueError("Coluna de valores não encontrada")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💰 Passo 5: Processar Valores

# COMMAND ----------

col_valor = mapeamento['valor']

print(f"\n🔄 PROCESSANDO COLUNA: {col_valor}")
print("=" * 80)

# Verificar tipo de dados
print(f"Tipo original: {df[col_valor].dtype}")

# Converter para numérico
if df[col_valor].dtype == 'object' or df[col_valor].dtype == 'string':
    print("Convertendo de string para numérico...")
    df[col_valor] = pd.to_numeric(df[col_valor], errors='coerce')
else:
    print("Já é numérico")

# Limpar dados
antes = len(df)
df = df[df[col_valor].notna()]
df = df[df[col_valor] > 0]
depois = len(df)

print(f"\n✅ LIMPEZA CONCLUÍDA")
print(f"   • Antes: {antes:,} registros")
print(f"   • Depois: {depois:,} registros")
print(f"   • Removidos: {antes-depois:,}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Passo 6: Estatísticas

# COMMAND ----------

print("\n💰 ESTATÍSTICAS DE VALORES")
print("=" * 80)
print(f"Total: R$ {df[col_valor].sum():,.2f}")
print(f"Média: R$ {df[col_valor].mean():,.2f}")
print(f"Mediana: R$ {df[col_valor].median():,.2f}")
print(f"Mínimo: R$ {df[col_valor].min():,.2f}")
print(f"Máximo: R$ {df[col_valor].max():,.2f}")
print("=" * 80)

# COMMAND ----------

# Distribuições
if 'uf' in mapeamento:
    print("\n🏆 TOP 10 ESTADOS:")
    display(df[mapeamento['uf']].value_counts().head(10).to_frame('Quantidade'))

# COMMAND ----------

if 'partido' in mapeamento:
    print("\n🏆 TOP 10 PARTIDOS:")
    display(df[mapeamento['partido']].value_counts().head(10).to_frame('Quantidade'))

# COMMAND ----------

if 'cargo' in mapeamento:
    print("\n🏆 DISTRIBUIÇÃO POR CARGO:")
    display(df[mapeamento['cargo']].value_counts().to_frame('Quantidade'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Passo 7: Visualizações

# COMMAND ----------

# Importar matplotlib aqui para evitar conflitos
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
sns.set_palette("husl")

print("=" * 80)
print("CRIANDO VISUALIZAÇÕES")
print("=" * 80)

# COMMAND ----------

# GRÁFICO 1: Histograma
print("\n📊 [1/5] Histograma de Valores")

# Calcular percentil 95 para filtro
p95 = df[col_valor].quantile(0.95)
valores_filtrados = df[df[col_valor] <= p95][col_valor]

print(f"   • Valores até percentil 95: R$ {p95:,.2f}")
print(f"   • Registros no gráfico: {len(valores_filtrados):,}")

fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(valores_filtrados, bins=50, color='coral', edgecolor='darkred', alpha=0.7)

ax.set_xlabel('Valor da Despesa (R$)', fontsize=12, weight='bold')
ax.set_ylabel('Frequência', fontsize=12, weight='bold')
ax.set_title(f'Distribuição de Valores (até R$ {p95:,.2f})\nEleições 2024', 
             fontsize=14, weight='bold')
ax.grid(axis='y', alpha=0.3)

mediana = valores_filtrados.median()
media = valores_filtrados.mean()
ax.axvline(mediana, color='blue', linestyle='--', linewidth=2, 
           label=f'Mediana: R$ {mediana:,.2f}')
ax.axvline(media, color='red', linestyle='--', linewidth=2, 
           label=f'Média: R$ {media:,.2f}')
ax.legend()

plt.tight_layout()
display(fig)
plt.close()

print("✅ Gráfico exibido\n")

# COMMAND ----------

# GRÁFICO 2: Top Estados
if 'uf' in mapeamento:
    print("📊 [2/5] Top 15 Estados")
    
    despesas = df.groupby(mapeamento['uf'])[col_valor].sum().sort_values(ascending=True).tail(15)
    despesas_m = despesas / 1_000_000
    
    print(f"   • Estados no gráfico: {len(despesas)}")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.barh(despesas_m.index, despesas_m.values, color='steelblue', edgecolor='navy')
    
    for i, (idx, val) in enumerate(despesas_m.items()):
        ax.text(val * 1.02, i, f'R$ {val:.1f}M', va='center', fontsize=9, weight='bold')
    
    ax.set_xlabel('Valor Total (Milhões R$)', fontsize=12, weight='bold')
    ax.set_ylabel('Estado', fontsize=12, weight='bold')
    ax.set_title('Top 15 Estados - Volume de Despesas\nEleições 2024', 
                 fontsize=14, weight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    display(fig)
    plt.close()
    
    print("✅ Gráfico exibido\n")

# COMMAND ----------

# GRÁFICO 3: Top Partidos
if 'partido' in mapeamento:
    print("📊 [3/5] Top 10 Partidos")
    
    despesas = df.groupby(mapeamento['partido'])[col_valor].sum().sort_values(ascending=False).head(10)
    despesas_m = despesas / 1_000_000
    
    print(f"   • Partidos no gráfico: {len(despesas)}")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(despesas_m)))
    bars = ax.bar(range(len(despesas_m)), despesas_m.values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Adicionar valores ACIMA das barras com espaçamento adequado
    max_val = despesas_m.max()
    for i, val in enumerate(despesas_m.values):
        ax.text(i, val + (max_val * 0.03), f'R$ {val:.1f}M', 
                ha='center', va='bottom', fontsize=10, weight='bold')
    
    # Configurar eixo X com nomes dos partidos
    ax.set_xticks(range(len(despesas_m)))
    ax.set_xticklabels(despesas_m.index, rotation=0, ha='center', fontsize=11, weight='bold')
    
    ax.set_xlabel('Partido', fontsize=13, weight='bold', labelpad=10)
    ax.set_ylabel('Valor Total (Milhões R$)', fontsize=13, weight='bold', labelpad=10)
    ax.set_title('Top 10 Partidos - Volume de Despesas\nEleições Municipais 2024', 
                 fontsize=16, weight='bold', pad=20)
    
    # Ajustar limites do eixo Y para dar espaço aos valores
    ax.set_ylim(0, max_val * 1.15)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    display(fig)
    plt.close()
    
    print("✅ Gráfico exibido\n")

# COMMAND ----------

# GRÁFICO 4: Pizza - Distribuição por Cargo
if 'cargo' in mapeamento:
    print("📊 [4/5] Distribuição por Cargo")
    
    cargo_counts = df[mapeamento['cargo']].value_counts()
    
    print(f"   • Cargos no gráfico: {len(cargo_counts)}")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    ax.pie(cargo_counts.values, labels=cargo_counts.index, autopct='%1.1f%%',
           startangle=90, colors=colors[:len(cargo_counts)], shadow=True,
           textprops={'fontsize': 10, 'weight': 'bold'})
    
    ax.set_title('Distribuição por Cargo\nEleições 2024', 
                 fontsize=14, weight='bold')
    
    plt.tight_layout()
    display(fig)
    plt.close()
    
    print("✅ Gráfico exibido\n")

# COMMAND ----------

# GRÁFICO 5: Boxplot por Cargo
if 'cargo' in mapeamento:
    print("📊 [5/5] Boxplot por Cargo")
    
    # Usar percentil 90 para filtro
    p90 = df[col_valor].quantile(0.90)
    df_filtrado = df[df[col_valor] <= p90].copy()
    
    print(f"   • Limite de valores: R$ {p90:,.2f}")
    print(f"   • Registros no gráfico: {len(df_filtrado):,}")
    
    if len(df_filtrado) > 0:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Criar boxplot manualmente para cada cargo
        cargos = df_filtrado[mapeamento['cargo']].unique()
        dados_por_cargo = [df_filtrado[df_filtrado[mapeamento['cargo']] == cargo][col_valor].values 
                           for cargo in cargos]
        
        bp = ax.boxplot(dados_por_cargo, labels=cargos, patch_artist=True)
        
        # Colorir boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors * (len(cargos)//len(colors) + 1)):
            patch.set_facecolor(color)
        
        ax.set_xlabel('Cargo', fontsize=12, weight='bold')
        ax.set_ylabel('Valor da Despesa (R$)', fontsize=12, weight='bold')
        ax.set_title(f'Distribuição de Valores por Cargo (até R$ {p90:,.2f})\nEleições 2024', 
                     fontsize=14, weight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        display(fig)
        plt.close()
        
        print("✅ Gráfico exibido\n")
    else:
        print("⚠️  DataFrame filtrado vazio, pulando boxplot\n")

# COMMAND ----------

print("=" * 80)
print("✅ TODAS AS VISUALIZAÇÕES CONCLUÍDAS")
print("=" * 80)

# COMMAND ----------

if 'candidato' in mapeamento:
    print("\n🤖 MACHINE LEARNING - CLUSTERING")
    print("=" * 80)
    
    # Criar dataset agregado
    print("Agregando dados por candidato...")
    
    colunas_grupo = [mapeamento['candidato']]
    if 'uf' in mapeamento:
        colunas_grupo.append(mapeamento['uf'])
    
    candidatos = df.groupby(colunas_grupo).agg({
        col_valor: ['sum', 'mean', 'count']
    }).reset_index()
    
    candidatos.columns = colunas_grupo + ['total_despesas', 'media_despesas', 'num_despesas']
    
    print(f"✅ {len(candidatos):,} candidatos únicos")
    
    # Mostrar amostra
    print("\n📋 Amostra dos dados agregados:")
    display(candidatos.head(10))
    
    # Clustering simples
    print("\nAplicando clustering...")
    
    # Importar sklearn aqui
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    X = candidatos[['total_despesas', 'num_despesas']].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    candidatos['cluster'] = kmeans.fit_predict(X_scaled)
    
    print(f"✅ Clustering concluído!")
    
    # Estatísticas por cluster
    print("\n📊 Estatísticas por Cluster:")
    stats = candidatos.groupby('cluster').agg({
        'total_despesas': ['count', 'mean', 'median'],
        'num_despesas': 'mean'
    }).round(2)
    display(stats)
    
    # Visualizar clusters
    print("\n📊 Visualização dos Clusters:")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for i in range(4):
        cluster_data = candidatos[candidatos['cluster'] == i]
        ax.scatter(cluster_data['num_despesas'], cluster_data['total_despesas'],
                   c=colors[i], label=f'Cluster {i} (n={len(cluster_data):,})',
                   alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Número de Despesas', fontsize=12, weight='bold')
    ax.set_ylabel('Total de Despesas (R$)', fontsize=12, weight='bold')
    ax.set_title('Clusters de Candidatos\nEleições 2024', 
                 fontsize=14, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    display(fig)
    plt.close()
    
    print("✅ Clustering visualizado")
    print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎉 Resumo Final

# COMMAND ----------

print("\n" + "=" * 80)
print("🎉 ANÁLISE CONCLUÍDA COM SUCESSO!")
print("=" * 80)

print(f"\n📊 RESUMO:")
print(f"   • Registros analisados: {len(df):,}")
print(f"   • Valor total: R$ {df[col_valor].sum():,.2f}")
print(f"   • Valor médio: R$ {df[col_valor].mean():,.2f}")
print(f"   • Colunas detectadas: {len(mapeamento)}")

graficos = 1  # histograma
if 'uf' in mapeamento:
    graficos += 1
if 'partido' in mapeamento:
    graficos += 1
if 'cargo' in mapeamento:
    graficos += 2

print(f"\n📊 VISUALIZAÇÕES:")
print(f"   • Gráficos criados: {graficos}")
print(f"   • Todos exibidos no notebook")

if 'candidato' in mapeamento:
    print(f"\n🤖 MACHINE LEARNING:")
    print(f"   • Candidatos únicos: {len(candidatos):,}")
    print(f"   • Clusters: 4")

print("\n" + "=" * 80)
print("✅ PROJETO FINALIZADO!")
print("=" * 80)

# COMMAND ----------


