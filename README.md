# Manutenção Preditiva IA

## Instalação

```bash
# 1. Cria o ambiente virtual chamado '.venv'
python3 -m venv .venv

# 2. Ativa o ambiente virtual no terminal atual
source .venv/bin/activate

# 3. Atualiza o gerenciador de pacotes interno da venv
pip install --upgrade pip

# 4. Instala todas as dependências na versão correta
pip install -r requirements.txt
````

## Fase 1: Análise Exploratória (EDA)

* **Inspeção dos dados:** Apresentesentamos as dimensões do dataset (número de linhas e colunas), os tipos de dados das variáveis e o resumo estatístico descritivo das colunas numéricas via método “.describe()”.

* **Gráficos exploratórios:** Geração e exportação automatizada de 4 visualizações analíticas para a pasta `outputs/plots/` utilizando as bibliotecas Matplotlib e Seaborn:
  * **Gráfico 1 (Histogramas de Distribuição):** Análise do perfil probabilístico e escalas das 5 variáveis preditoras contínuas (sensores).
  * **Gráfico 2 (Bar Plot de Desbalanceamento):** Quantificação estrita da proporção entre a classe majoritária (Operação Normal - 0) e a minoritária (Falha Mecânica - 1) na variável alvo principal `falha_maquina`.
  * **Gráfico 3 (Heatmap de Correlação de Pearson):** Avaliação de colinearidade focada exclusivamente entre os sensores reais e o target, descartando as colunas de motivos técnicos específicos (`falha_twf`, `falha_hdf`, etc.) para evitar o vazamento de dados (*data leakage*).
  * **Gráfico 4 [BÔNUS] (Dispersão Física de Operação):** Cruzamento dinâmico entre `velocidade_rotacao_rpm` e `torque_nm` segregados pelo estado de quebra, 
  mapeando visualmente a fronteira física de restrição operacional da máquina.
  * **Restrições das Notas de Engenharia (Conformidade de Dados):** * As colunas de diagnósticos específicos (`falha_twf`, `falha_hdf`, `falha_pwf`, `falha_osf`, `falha_rnf`) foram **excluídas do mapeamento de correlação** e de qualquer futuro vetor de atributos ($X$). 
  * Por representarem o histórico de pós-evento (motivo da quebra), sua inclusão causaria um viés de *data leakage* (vazamento de rótulo), invalidando a capacidade preditiva do modelo em tempo real. Elas permanecem no ecossistema local apenas para fins de auditoria e consulta.

* **Interpretação dos resultados:** Insira uma célula de texto analisando os valores numéricos e os padrões identificados nos gráficos, explicitando como eles direcionam a estratégia de modelagem.
