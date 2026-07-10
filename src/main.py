"""
Pipeline de IA para Análise Preditiva na Indústria 4.0
Fase 1: Análise Exploratória de Dados (EDA) - Inspeção dos Dados, Gráficos exploratórios, Interpretação dos resultados
Fase 2: Limpeza e Tratamento de Dados (Data Prep) - Limpeza e Estruturação de Dados, Tratamento de Dados Ausentes, Diagnóstico de Outliers
Fase 3: Feature Engineering - Criar Features
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==============================================================================
# FASE 1: ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
# ==============================================================================

#-------------------------------------------------------------------------------
# Inspeção dos Dados
#-------------------------------------------------------------------------------

def inspecionar_dados(df):
    """
    Fase 1 (Tópico 1) - Realiza a inspeção estrutural e exibe 
    o resumo estatístico descritivo inicial da base de dados.
    """
    print("\n" + "=" * 60)
    print("=== FASE 1: INSPEÇÃO DOS DADOS ===")
    print("=" * 60)

    # 1. Dimensões do Dataset (Linhas e Colunas)
    print(f"\n[ESTRUTURA] Dimensões do Dataset (Linhas, Colunas): {df.shape}")
    print(f"[ESTRUTURA] Lista de colunas: {list(df.columns)}")

    # 2. Tipos de Dados de cada variável
    print("\n--- TIPOS DE DADOS DAS VARIÁVEIS ---")
    print(df.dtypes)

    # 3. Contagem de Valores Nulos (Mapeamento prévio)
    print("\n--- VALORES NULOS DETECTADOS POR COLUNA ---")
    print(df.isnull().sum())

    # 4. Primeiros registros para validação visual
    print("\n--- VISUALIZAÇÃO DOS PRIMEIROS REGISTROS (HEAD) ---")
    print(df.head().to_string())

    # 5. Resumo Estatístico Descritivo (.describe())
    print("\n--- RESUME ESTATÍSTICO DESCRITIVO (.describe()) ---")
    print(df.describe().to_string())
    print("=" * 60 + "\n")

#-------------------------------------------------------------------------------
# Gráficos exploratórios
#-------------------------------------------------------------------------------

def gerar_graficos(df):
    """
    Fase 1 (Tópico 2) - Gera e salva os 3 gráficos obrigatórios do edital 
    mais um 4º gráfico focado em diagnóstico de falhas.
    """
    print("\n" + "=" * 60)
    print("=== FASE 1: GERAÇÃO DOS GRÁFICOS EXPLORATÓRIOS ===")
    print("=" * 60)

    # Caminho para salvar os gráficos
    path_graphs = "../outputs/plots"
    
    # Configurações globais de estilo
    sns.set_theme(style="whitegrid")

    # Listamos os 5 sensores operacionais. Deixamos de fora os IDs e as falhas específicas, cumprindo a primeira restrição do Departamento de Engenharia
    colunas_sensores = [
        'temperatura_ar_k', 'temperatura_processo_k', 
        'velocidade_rotacao_rpm', 'torque_nm', 'desgaste_ferramenta_min'
    ]

    # --------------------------------------------------------------------------
    # GRÁFICO 1: Histograma de Distribuição das Variáveis
    # --------------------------------------------------------------------------
    """
    Este gráfico mostra a distribuição de cada sensor. 
    Como as escalas são muito diferentes — RPM está na casa dos milhares e Torque nas dezenas —, 
    isso mostra que o KNN vai precisar de um escalonamento de dados mais para frente   
    """

    print("[GERANDO GRÁFICO] Plotando histogramas das variáveis dos sensores...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(colunas_sensores):
        sns.histplot(data=df, x=col, kde=True, ax=axes[i], color="#2196F3")
        axes[i].set_title(f"Distribuição: {col.replace('_', ' ').title()}")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Frequência")

    fig.delaxes(axes[-1])
    plt.suptitle("Distribuição Estatística das Variáveis dos Sensores Industriais", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    caminho_g1 = os.path.join(path_graphs, "eda_distribuicao_sensores.png")
    plt.savefig(caminho_g1, dpi=150)
    plt.close()
    print(f"  [OK] Gráfico 1 salvo com sucesso em: {caminho_g1}")

    # --------------------------------------------------------------------------
    # GRÁFICO 2: Taxa de Desbalanceamento do Alvo
    # --------------------------------------------------------------------------
    print("[GERANDO GRÁFICO] Plotando taxa de desbalanceamento da classe alvo...")

    plt.figure(figsize=(7, 5))
    
    ax = sns.countplot(data=df, x='falha_maquina', palette="Set1", hue='falha_maquina', legend=False)
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=4, fontweight='bold')
        
    plt.title("Proporção e Desbalanceamento da Variável Alvo (Falha)", fontsize=12, fontweight='bold')
    plt.xlabel("Estado Operacional (0 = Normal, 1 = Falha Mecânica)")
    plt.ylabel("Contagem Absoluta de Registros")
    plt.xticks([0, 1], ["Operação Normal (0)", "Avaria / Quebra (1)"])
    plt.tight_layout()
    
    caminho_g2 = os.path.join(path_graphs, "eda_desbalanceamento_alvo.png")
    plt.savefig(caminho_g2, dpi=150)
    plt.close()
    print(f"  [OK] Gráfico 2 salvo com sucesso em: {caminho_g2}")

    # --------------------------------------------------------------------------
    # GRÁFICO 3: Mapa de Calor (Correlação de Pearson)
    # --------------------------------------------------------------------------
    """
    A matriz só cruza os sensores com o alvo principal (falha_maquina). 
    Excluímos as falhas secundárias para evitar o vazamento de dados, 
    garantindo que o modelo preveja o futuro e não o passado
    """
    print("[GERANDO GRÁFICO] Plotando mapa de calor da correlação linear...")
    plt.figure(figsize=(10, 8))
    
    df_corr = df[colunas_sensores + ['falha_maquina']].corr(method='pearson')
    sns.heatmap(df_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    
    plt.title("Matriz de Correlação de Pearson (Métricas Lineares vs Target)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    caminho_g3 = os.path.join(path_graphs, "eda_matriz_correlacao.png")
    plt.savefig(caminho_g3, dpi=150)
    plt.close()
    print(f"  [OK] Gráfico 3 salvo com sucesso em: {caminho_g3}")

    # --------------------------------------------------------------------------
    # GRÁFICO 4: Boxplot de Relação Mecânica (Torque vs Rotação por Falha)
    # --------------------------------------------------------------------------
    """
    Cruzamos RPM contra Torque. Como a potência depende dessas duas forças, 
    o gráfico colorindo as falhas mostra visualmente a fronteira física exata onde a máquina entra em sobrecarga e quebra.
    """
    print("[GERANDO GRÁFICO BÔNUS] Plotando dispersão de Torque e RPM segregados por Falha...")
    plt.figure(figsize=(9, 6))
    
    sns.scatterplot(data=df.dropna(), x='velocidade_rotacao_rpm', y='torque_nm', 
                    hue='falha_maquina', palette="coolwarm", alpha=0.6, style='falha_maquina')
    
    plt.title("Análise Física Interativa: Velocidade de Rotação (RPM) vs Torque (Nm)", fontsize=12, fontweight='bold')
    plt.xlabel("Velocidade de Rotação (RPM)")
    plt.ylabel("Torque (Nm)")
    plt.legend(title="Estado", labels=["Normal (0)", "Falha (1)"])
    plt.tight_layout()
    
    caminho_g4 = os.path.join(path_graphs, "eda_bonus_relacao_mecanica.png")
    plt.savefig(caminho_g4, dpi=150)
    plt.close()
    print(f"  [OK] Gráfico 4 (Bônus) salvo com sucesso em: {caminho_g4}")
    print("=" * 60 + "\n")

# ==============================================================================
# FASE 2: LIMPEZA E TRATAMENTO DE DADOS (DATA PREP)
# ==============================================================================

# --------------------------------------------------------------------------
# 1. Limpeza e Estruturação de Dados
# --------------------------------------------------------------------------
def limpar_dados(df_raw):
    """Fase 2 (Tópico 1) - Identifique e remova as linhas duplicadas e inconsistentes."""
    print("[INFO] Iniciando limpeza: Duplicatas, formatação de texto e filtros físicos...")
    
    # Criamos uma cópia para proteger a integridade do dado extraído
    df = df_raw.copy()
    n_inicial = len(df)
    relatorio = {}

    # 1. Identificamos e removemos possíveis linhas duplicadas pelo ID único (udi)
    n_antes_dup = len(df)
    df = df.drop_duplicates(subset=["udi"])
    relatorio["duplicatas_removidas"] = n_antes_dup - len(df)

    # 2. Padronização de Texto (Removemos espaços em branco e deixar em maiúsculo)
    for col in ["id_produto", "tipo"]:
        df[col] = df[col].astype(str).str.strip().str.upper()

    # 3. Filtro, pois grandezas não podem ser negativas ou zero absolutas
    n_antes_inv = len(df)
    df = df[(df["velocidade_rotacao_rpm"] > 0) & 
            (df["temperatura_ar_k"] > 0) & 
            (df["temperatura_processo_k"] > 0) & 
            (df["torque_nm"] > 0)]

    relatorio["anomalias_fisicas_removidas"] = n_antes_inv - len(df)

    # 4. Garantimos tipos de dados numéricos corretos
    df["desgaste_ferramenta_min"] = df["desgaste_ferramenta_min"].astype(int)

    # Métricas Finais do Relatório
    n_final = len(df)
    relatorio["registros_iniciais"] = n_inicial
    relatorio["registros_finais"] = n_final
    relatorio["registros_removidos_total"] = n_inicial - n_final

    print("\n=== RELATÓRIO DE LIMPEZA (TÓPICO 1) ===")
    for chave, valor in relatorio.items():
        nome_formatado = chave.replace("_", " ").title()
        print(f"  {nome_formatado}: {valor}")
    print("========================================\n")

    return df, relatorio

# --------------------------------------------------------------------------
# 2. Tratamento de Dados Ausentes (Imputação por Média/Mediana)
# --------------------------------------------------------------------------
def imputar_dados(df_limpo):
    """Fase 2 (Tópico 2) - Identifica e imputa valores ausentes com justificativa técnica."""
    print("[INFO] Iniciando imputação de dados ausentes...")

    OUTPUT_DIR = Path("../outputs")
    
    df = df_limpo.copy()
    nulos_antes = df.isnull().sum()
    relatorio = {}

    # - Torque: Média (Normal)
    # - RPM e Temperaturas: Mediana (Assimetria/Robustez)
    
    colunas_media = ["torque_nm"]
    colunas_mediana = ["velocidade_rotacao_rpm", "temperatura_ar_k", "temperatura_processo_k"]

    for col in colunas_media:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
            
    for col in colunas_mediana:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Auditoria
    nulos_depois = df.isnull().sum().sum()
    relatorio["total_nulos_imputados"] = int(nulos_antes.sum() - nulos_depois)
    
    print("\n=== RELATÓRIO DE IMPUTAÇÃO (TÓPICO 2) ===")
    print(f"  Valores ausentes tratados: {relatorio['total_nulos_imputados']}")
    print("  Estratégia: Torque (Média), Demais (Mediana)")
    print("========================================\n")

    caminho_csv = OUTPUT_DIR / "dados_limpos.csv"
    
    # Salvando o dataframe limpo
    df.to_csv(caminho_csv, index=False)
    print(f"[INFO] Dados limpos salvos com sucesso em: {caminho_csv}")

    return df, relatorio

# --------------------------------------------------------------------------
# 3. Diagnóstico de Outliers (Visualização via Boxplots)
# --------------------------------------------------------------------------
def gerar_boxplots(df):
    """Fase 2 (Tópico 3) - Gera boxplots para identificar outliers nas variáveis explicativas."""
    print("[INFO] Gerando diagnóstico visual de outliers (Boxplots)...")
    
    path_graphs = "../outputs/plots"
    colunas_sensores = [
        'temperatura_ar_k', 'temperatura_processo_k', 
        'velocidade_rotacao_rpm', 'torque_nm', 'desgaste_ferramenta_min'
    ]
    
    # Criar subplots para os 5 sensores
    fig, axes = plt.subplots(1, 5, figsize=(20, 6))
    sns.set_theme(style="whitegrid")
    
    for i, col in enumerate(colunas_sensores):
        sns.boxplot(y=df[col], ax=axes[i], color="#FF9800")
        axes[i].set_title(col.replace('_', ' ').title())
        axes[i].set_ylabel("")
        
    plt.suptitle("Diagnóstico de Outliers: Variáveis Explicativas (Sensores)", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    caminho_bp = os.path.join(path_graphs, "prep_diagnostico_outliers.png")
    plt.savefig(caminho_bp, dpi=150)
    plt.close()
    
    print(f"  [OK] Gráfico de Boxplots salvo em: {caminho_bp}")

# ==============================================================================
# FASE 3: FEATURE ENGINEERING  
# ==============================================================================

# --------------------------------------------------------------------------
# 1. Criando Features 
# --------------------------------------------------------------------------

def criar_features(df_imputado):
    """Fase 3 - Criação de novas variáveis baseadas em lógica física."""
    print("[INFO] Iniciando Feature Engineering...")
    
    df = df_imputado.copy()
    
    # Exemplo: Criar métrica de intensidade de desgaste por temperatura de processo
    # Adicionamos uma pequena constante para evitar divisão por zero (já tratamos nulos antes!)
    df["taxa_desgaste_processo"] = df["desgaste_ferramenta_min"] / (df["temperatura_processo_k"] + 1e-6)
    
    # Criar uma flag de sobrecarga térmica (Booleana convertida para int)
    df["flag_sobrecarga_termica"] = (df["temperatura_processo_k"] > df["temperatura_processo_k"].median()).astype(int)

    print("[INFO] Novas features 'taxa_desgaste_processo' e 'flag_sobrecarga_termica' criadas.")
    
    # Salvando a nova versão
    caminho_csv = "../outputs/dados_enriquecidos.csv"
    df.to_csv(caminho_csv, index=False)
    
    return df

# ==============================================================================
# MAIN
# ==============================================================================
def main():    
    data_path = "../data/manutencao_preditiva.csv"
    
    try:
        print(f"[INFO] Carregando a base de dados: {data_path}")
        df_raw = pd.read_csv(data_path)
        
        #---------------------------------------------------------
        # Fase 1: Análise Exploratória (EDA)
        #---------------------------------------------------------
        # Inspeção dos dados
        inspecionar_dados(df_raw)

        # Gráficos Exploratórios
        gerar_graficos(df_raw)

        #---------------------------------------------------------
        # Fase 2: Limpeza e Tratamento de Dados (Data Prep)
        #---------------------------------------------------------
        # Limpeza e Estruturação de Dados
        df_limpo, relatorio_limpeza = limpar_dados(df_raw)

        # Tratamento de Dados Ausentes
        df_imputado, relatorio_imputacao = imputar_dados(df_limpo)

        # Diagnóstico de Outliers
        gerar_boxplots(df_imputado)

        #---------------------------------------------------------
        # Fase 3: Feature Engineering
        #---------------------------------------------------------
        # Criando Features
        df_enriquecido = criar_features(df_imputado)
        
    except FileNotFoundError:
        print(f"[ERRO] O arquivo não foi encontrado.")

if __name__ == "__main__":
    main()