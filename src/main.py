"""
Pipeline de IA para Análise Preditiva na Indústria 4.0
Fase 1: Análise Exploratória de Dados (EDA) - Inspeção dos Dados
"""

import os
import pandas as pd

# ==============================================================================
# FASE 1: ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
# ==============================================================================

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

# ==============================================================================
# MAIN
# ==============================================================================
def main():    
    data_path = "../data/manutencao_preditiva.csv"
    
    try:
        print(f"[INFO] Carregando a base de dados: {data_path}")
        df_raw = pd.read_csv(data_path)
        
        # Fase 1: Análise Exploratória (EDA)/Inspeção dos dados
        inspecionar_dados(df_raw)
        
    except FileNotFoundError:
        print(f"[ERRO] O arquivo não foi encontrado.")

if __name__ == "__main__":
    main()