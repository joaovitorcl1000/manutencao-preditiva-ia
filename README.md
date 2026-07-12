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
```

## Fase 1: Análise Exploratória (EDA)

* **Inspeção dos dados:** Apresentamos as dimensões do dataset (número de linhas e colunas), os tipos de dados das variáveis e o resumo estatístico descritivo das colunas numéricas via método “.describe()”. Os resultados do terminal podem ser vistos ao fim deste arquivo em 'Execução e Diagnóstico Local (Console Output)'. 

* **Gráficos exploratórios:** Geração e exportação automatizada de 4 visualizações analíticas para a pasta `outputs/plots/` utilizando as bibliotecas Matplotlib e Seaborn. Os resultados também podem ser vistos ao fim deste arquivo em 'Execução e Diagnóstico Local (Console Output)'. Temos
  * **Gráfico 1 (Histogramas de Distribuição):** Análise do perfil probabilístico e escalas das 5 variáveis preditoras contínuas (sensores).
  * **Gráfico 2 (Bar Plot de Desbalanceamento):** Quantificação estrita da proporção entre a classe majoritária (Operação Normal - 0) e a minoritária (Falha Mecânica - 1) na variável alvo principal `falha_maquina`.
  * **Gráfico 3 (Heatmap de Correlação de Pearson):** Avaliação de colinearidade focada exclusivamente entre os sensores reais e o target, descartando as colunas de motivos técnicos específicos (`falha_twf`, `falha_hdf`, etc.) para evitar o vazamento de dados (*data leakage*).
  * **Gráfico 4 (Dispersão Física de Operação):** Cruzamento dinâmico entre `velocidade_rotacao_rpm` e `torque_nm` segregados pelo estado de quebra, 
  mapeando visualmente a fronteira física de restrição operacional da máquina.
  * **Restrições das Notas de Engenharia (Conformidade de Dados):** * As colunas de diagnósticos específicos (`falha_twf`, `falha_hdf`, `falha_pwf`, `falha_osf`, `falha_rnf`) foram **excluídas do mapeamento de correlação** e de qualquer futuro vetor de atributos ($X$). 
  * Por representarem o histórico de pós-evento (motivo da quebra), sua inclusão causaria um viés de *data leakage* (vazamento de rótulo), invalidando a capacidade preditiva do modelo em tempo real. Elas permanecem no ecossistema local apenas para fins de auditoria e consulta.

* **Interpretação dos resultados:** Analisamos os valores numéricos e os padrões identificados nos gráficos, explicitando como eles direcionam a estratégia de modelagem. Os resultados também podem ser vistos ao fim deste arquivo em 'Execução e Diagnóstico Local (Console Output)'.

## Fase 2: Limpeza e Tratamento de Dados (Data Prep)

* **Limpeza e Estruturação de Dados:** O script implementa a função `limpar_dados`, que atua como um funil rigoroso de qualidade para garantir a integridade física e matemática do dataset. Suas operações executam sequencialmente:

    * **Remoção de Duplicatas:** Mapeia e elimina registros redundantes utilizando a coluna `udi` (identificador único da peça/motor) como chave de referência restrita.
    * **Padronização Categórica:** Remove espaços em branco residuais (`.strip()`) e uniformiza as strings em letras maiúsculas (`.upper()`) nas colunas identificadoras `id_produto` e `tipo`, garantindo consistência para as próximas fases.
    * **Filtro de Conformidade Física:** Executa uma limpeza rigorosa de anomalias, excluindo da base quaisquer registros que apresentem leituras fisicamente impossíveis para um motor industrial (ex: velocidade de rotação, temperaturas ou torque menores ou iguais a zero). 
    * **Consistência de Tipos (*Type Casting*):** Força a tipagem correta da variável `desgaste_ferramenta_min` para o formato inteiro (`int`), prevenindo erros de alocação de memória e garantindo precisão numérica.
    * **Auditoria de Pipeline:** A função gera um relatório transacional no console que rastreia quantitativamente cada linha alterada ou removida, assegurando transparência no tratamento inicial da base de 10.000 registros.

* **Tratamento de Dados Ausentes:** Foi realizada a identificação de valores nulos no conjunto de dados e aplicada a imputação utilizando métricas de tendência central, com a estratégia selecionada conforme a distribuição de cada variável observada na Análise Exploratória (Fase 1):

    * **Torque (Nm):** Utilizou-se a **Média**, visto que a variável exibe uma distribuição Gaussiana (Normal) simétrica, onde a média representa o valor central com maior precisão estatística.
    * **Velocidade (RPM) e Temperaturas (Ar/Processo):** Utilizou-se a **Mediana**, devido à presença de assimetria e valores extremos (*outliers*) identificados nos histogramas. A mediana assegura robustez, evitando que valores discrepantes distorçam o ponto central da distribuição.

* **Diagnóstico de Outliers:** Foram gerados gráficos do tipo *Boxplot* para as variáveis explicativas, revelando que, enquanto *Temperaturas* e *Desgaste* apresentam distribuições estáveis, as variáveis de *Velocidade (RPM)* e *Torque* concentram *outliers* significativos acima dos limites superiores. 

    * Essa distribuição não indica erros de medição, mas sim estados operacionais críticos: a presença desses valores extremos é a "assinatura" física de sobrecargas mecânicas e instabilidades, sendo essencial manter estes dados preservados para que o modelo aprenda a identificar padrões de falha.
    * O padrão observado nos *boxplots* confirma que a variabilidade dos sensores de *Torque* e *RPM* não é ruído aleatório, mas sim um reflexo da dinâmica operacional do equipamento em condições de esforço. Conclui-se, portanto, que a retenção dos *outliers* é indispensável para o treinamento de um modelo preditivo eficaz, visto que eles representam os limites físicos de operação onde a falha mecânica torna-se estatisticamente provável.

## Fase 3: Feature Engineering

* **Criando Features:** Criamos uma nova coluna numérica por meio de operação matemática entre colunas existentes, tratando os valores nulos previamente. Geramos de novas colunas numéricas baseadas em relações físicas entre sensores existentes.
    * **Implementação:** * `taxa_desgaste_processo`: Calculada via divisão entre `desgaste_ferramenta_min` e `temperatura_processo_k`.
        * `flag_sobrecarga_termica`: Variável binária baseada na mediana da temperatura.
    * **Tratamento:** As operações foram realizadas após a etapa de imputação de nulos da Fase 2, garantindo que nenhum valor `NaN` propagasse erro na divisão matemática.
    * **Output:** Dados enriquecidos salvos em `outputs/dados_enriquecidos.csv`.
    * **Potência Estimada (`potencia_estimada`):** Calculada pela multiplicação entre `velocidade_rotacao_rpm` e `torque_nm`. Representa a potência mecânica do sistema. Variações anômalas nesta métrica, mesmo com RPM constante, indicam falhas iminentes no esforço do motor ou resistência excessiva no corte.

## Fase 4: Divisão e Balanceamento dos Dados

* **Variáveis Preditoras e Alvo:** Isolar as variáveis preditoras da variável alvo.
    * **Ação:** O dataset foi particionado em:
        * `X`: Matriz contendo todas as features de sensores e as novas variáveis derivadas (taxa de desgaste, flag térmica e potência).
        * `y`: Vetor contendo a variável binária `falha_maquina`.
    * **Limpeza:** Colunas de identificação (`udi`, `id_produto`) e categóricas (`tipo`) foram descartadas desta etapa para evitar ruído no modelo preditivo.

* **Pareto:** Dividimos os dados seguindo o princípio de Pareto, em treino (80%) e teste (20%) utilizando o parâmetro stratify=y.

* **Reamostragem:** Aplicamos uma técnica de reamostragem (SMOTE ou Random Under Sampling) exclusivamente nos dados de treino para evitar o vazamento de dados (Data Leakage).

## Fase 5: Escalonamento de Variáveis (StandardScaler)

* **Abordagem Híbrida:** Aplicamos o StandardScaler apenas nas variáveis contínuas destinadas ao modelo KNN (utilizando fit_transform no treino e transform no teste).
    * **Para KNN (Modelos de Distância):** Aplicação de `StandardScaler` nas variáveis contínuas para equalizar a influência dos sensores no cálculo da distância Euclidiana.
    * **Para Árvores de Decisão:** Os dados foram mantidos sem escalonamento. Como o algoritmo de árvore realiza partições baseadas em limiares (splits) de valores, a normalização é desnecessária e mantê-la preserva a interpretabilidade das features originais.
* **Segurança:** Utilizado `fit_transform` exclusivamente no treino para evitar *data leakage*. Mantemos os dados da Árvore de Decisão sem escalonamento, justificando no código o motivo de o algoritmo ser imune à escala dos atributos.




# Execução e Diagnóstico Local (Console Output)

## Resultados Fase 1: Análise Exploratória (EDA)

### Inspeção dos dados

```
[ESTRUTURA] Dimensões do Dataset (Linhas, Colunas): (10000, 14)
[ESTRUTURA] Lista de colunas: ['udi', 'id_produto', 'tipo', 'temperatura_ar_k', 'temperatura_processo_k', 'velocidade_rotacao_rpm', 'torque_nm', 'desgaste_ferramenta_min', 'falha_maquina', 'falha_twf', 'falha_hdf', 'falha_pwf', 'falha_osf', 'falha_rnf']

--- TIPOS DE DADOS DAS VARIÁVEIS ---
udi                          int64
id_produto                  object
tipo                        object
temperatura_ar_k           float64
temperatura_processo_k     float64
velocidade_rotacao_rpm     float64
torque_nm                  float64
desgaste_ferramenta_min      int64
falha_maquina                int64
falha_twf                    int64
falha_hdf                    int64
falha_pwf                    int64
falha_osf                    int64
falha_rnf                    int64
dtype: object

--- VALORES NULOS DETECTADOS POR COLUNA ---
udi                          0
id_produto                   0
tipo                         0
temperatura_ar_k           500
temperatura_processo_k     500
velocidade_rotacao_rpm     500
torque_nm                  500
desgaste_ferramenta_min      0
falha_maquina                0
falha_twf                    0
falha_hdf                    0
falha_pwf                    0
falha_osf                    0
falha_rnf                    0
dtype: int64

--- VISUALIZAÇÃO DOS PRIMEIROS REGISTROS (HEAD) ---
   udi id_produto tipo  temperatura_ar_k  temperatura_processo_k  velocidade_rotacao_rpm  torque_nm  desgaste_ferramenta_min  falha_maquina  falha_twf  falha_hdf  falha_pwf  falha_osf  falha_rnf
0    1     M14860    M             298.1                   308.6                  1551.0       42.8                        0              0          0          0          0          0          0
1    2     L47181    L             298.2                   308.7                  1408.0       46.3                        3              0          0          0          0          0          0
2    3     L47182    L             298.1                   308.5                  1498.0       49.4                        5              0          0          0          0          0          0
3    4     L47183    L               NaN                     NaN                     NaN        NaN                        7              0          0          0          0          0          0
4    5     L47184    L             298.2                   308.7                  1408.0       40.0                        9              0          0          0          0          0          0

--- RESUME ESTATÍSTICO DESCRITIVO (.describe()) ---
               udi  temperatura_ar_k  temperatura_processo_k  velocidade_rotacao_rpm    torque_nm  desgaste_ferramenta_min  falha_maquina     falha_twf     falha_hdf     falha_pwf     falha_osf    falha_rnf
count  10000.00000       9500.000000             9500.000000             9500.000000  9500.000000             10000.000000   10000.000000  10000.000000  10000.000000  10000.000000  10000.000000  10000.00000
mean    5000.50000        300.002158              310.000895             1539.245263    39.974168               107.951000       0.033900      0.004600      0.011500      0.009500      0.009800      0.00190
std     2886.89568          2.001689                1.486432              180.273589     9.995453                63.654147       0.180981      0.067671      0.106625      0.097009      0.098514      0.04355
min        1.00000        295.300000              305.700000             1168.000000     3.800000                 0.000000       0.000000      0.000000      0.000000      0.000000      0.000000      0.00000
25%     2500.75000        298.300000              308.800000             1423.000000    33.100000                53.000000       0.000000      0.000000      0.000000      0.000000      0.000000      0.00000
50%     5000.50000        300.100000              310.100000             1504.000000    40.100000               108.000000       0.000000      0.000000      0.000000      0.000000      0.000000      0.00000
75%     7500.25000        301.500000              311.100000             1613.000000    46.700000               162.000000       0.000000      0.000000      0.000000      0.000000      0.000000      0.00000
max    10000.00000        304.500000              313.800000             2886.000000    76.600000               253.000000       1.000000      1.000000      1.000000      1.000000      1.000000      1.00000
============================================================
```

### Gráficos exploratórios

| Gráficos de Distribuição e Alvo | Matriz e Relação Física |
| :---: | :---: |
| **Gráfico 1: Distribuição**<br><img src="outputs/plots/eda_distribuicao_sensores.png" width="400" alt="Distribuição"> | **Gráfico 2: Desbalanceamento**<br><img src="outputs/plots/eda_desbalanceamento_alvo.png" width="400" alt="Desbalanceamento"> |
| **Gráfico 3: Correlação**<br><img src="outputs/plots/eda_matriz_correlacao.png" width="400" alt="Correlação"> | **Gráfico 4: Dispersão Física**<br><img src="outputs/plots/eda_relacao_mecanica.png" width="400" alt="Bônus"> |

### Interpretação dos Resultados: Inspeção Estrutural e Estatística

Com base na varredura inicial do dataset gerada no terminal, identificamos três padrões críticos que guiarão diretamente as próximas etapas de pré-processamento e modelagem da IA:

**1. Padrão de Ausência Síncrona de Dados (Telemetria):**
A função `.isnull().sum()` revelou exatamente **500 valores nulos** concentrados simultaneamente nas quatro colunas de sensores contínuos (`temperatura_ar_k`, `temperatura_processo_k`, `velocidade_rotacao_rpm` e `torque_nm`). O registro de índice 3 na visualização `.head()` confirma esse comportamento. 
* **Direcionamento para Modelagem:** Isso não é um erro aleatório, mas sim um indicativo de queda de rede ou falha no barramento de telemetria da máquina. Na etapa de preparação de dados, precisaremos aplicar técnicas de imputação (como preenchimento pela mediana ou K-NN Imputer) ou remover essas linhas de forma controlada para não quebrar o treinamento dos modelos.

**2. Discrepância Massiva de Escalas Físicas (Impacto Geométrico):**
O resumo `.describe()` expõe uma diferença gritante nas ordens de magnitude das grandezas físicas monitoradas. A velocidade de rotação opera na casa dos milhares (média de **1539 RPM**, máximo de **2886 RPM**), enquanto o torque atua nas dezenas (média de **39.9 Nm**). 
* **Direcionamento para Modelagem:** Como muitos algoritmos de Machine Learning (como K-NN, SVM ou Redes Neurais baseadas em gradiente) calculam distâncias espaciais ou otimizam pesos matemáticos, a grandeza do RPM dominaria completamente a função de custo, "apagando" a influência do torque. É **obrigatória** a aplicação de padronização matemática (ex: `StandardScaler`) no pipeline para colocar todas as variáveis na mesma escala estatística.

**3. Severidade do Desbalanceamento da Variável Alvo:**
A média da variável `falha_maquina` é de **0.0339**. Isso comprova que, de todo o histórico operacional (10.000 registros), apenas **3.39%** (339 linhas) representam eventos reais de falha mecânica. 
* **Direcionamento para Modelagem:** Trata-se de um problema clássico de detecção de anomalias raras. Se treinarmos um classificador sem tratar isso, ele atingirá 96.6% de acurácia apenas "chutando" que a máquina nunca quebra. Por isso, a Acurácia será descartada como métrica principal. O sucesso do projeto será medido pelo **F1-Score** e **Recall**, exigindo o uso de técnicas de reamostragem sintética (como o algoritmo SMOTE) para balancear a classe minoritária durante o treino.

**Gráfico 1: Morfologia e Comportamento Probabilístico dos Sensores (Distribuições):** A análise do Gráfico 1 revela que as variáveis preditoras operam sob regimes estatísticos e físicos completamente distintos. O **Torque** aproxima-se de uma distribuição Normal (Gaussiana) simétrica, indicando um processo mecânico estável centrado em sua média. O **Desgaste da Ferramenta** apresenta uma nítida distribuição Uniforme, o que traduz fielmente o ciclo de vida da peça na fábrica: um acúmulo linear de desgaste temporal (minutos) desde a instalação até o momento limite da troca. Já a **Velocidade de Rotação (RPM)** possui uma forte assimetria à direita (cauda longa), provando que picos extremos de rotação ocorrem, mas são eventos de exceção. Por fim, as **Temperaturas** (Ar e Processo) exibem perfis multimodais (vários picos de frequência), o que sugere que a máquina opera sob diferentes regimes térmicos, possivelmente devido a diferentes turnos, lotes de peças ou flutuações ambientais sazonais. Esta pluralidade morfológica (distribuições normais, uniformes, assimétricas e multimodais) consolida a tese de que algoritmos paramétricos estritos (que assumem normalidade global dos dados) terão dificuldade de generalização. Isso valida fortemente a adoção de algoritmos baseados em árvores de decisão (como Random Forest ou XGBoost). Tais modelos particionam o espaço vetorial de forma não linear, sendo naturalmente robustos e agnósticos tanto à escala quanto ao formato exato das distribuições subjacentes.

**Gráfico 2: Quantificação Visual do Desbalanceamento (Bar Plot da Variável Alvo):** O Gráfico 2 materializa a severidade da assimetria de classes já detectada na inspeção estatística. Observamos uma disparidade esmagadora na distribuição do target: 9.661 registros (96,61%) correspondem à operação normal do ativo, contra apenas 339 ocorrências (3,39%) de falhas mecânicas reais. Esta visualização crava a impossibilidade de treinar modelos preditivos com o dataset cru. Sem intervenção, qualquer algoritmo adotará um viés de classe majoritária para minimizar o erro global, alcançando falsos 96,61% de acurácia simplesmente classificando todos os cenários como "operação normal". Para forçar a IA a aprender os padrões de quebra, a aplicação de técnicas de reamostragem (como o SMOTE) ou o uso de penalidades de peso (*class_weight*) durante a Fase de Modelagem torna-se inegociável. Reafirma-se também a troca da Acurácia por métricas focadas no acerto da classe minoritária, como Recall e F1-Score.

**Gráfico 3: Colinearidade e Redundância Informacional (Matriz de Pearson):** O Gráfico 3 (Heatmap) mapeia as relações lineares do ecossistema da máquina e revela duas fortes zonas de multicolinearidade. Primeiro, há uma correlação positiva severa (**0.88**) entre a Temperatura do Ar e a Temperatura do Processo, indicando redundância termodinâmica. Segundo, confirma-se a fortíssima correlação negativa (**-0.88**) entre a Velocidade de Rotação e o Torque, reflexo direto da restrição de potência do motor. Além disso, notamos que a correlação linear isolada de qualquer sensor com a variável alvo `falha_maquina` é baixíssima (a mais forte é o Torque, com apenas **0.19**). A ausência de fortes correlações preditivas lineares comprova que as falhas são fenômenos complexos, gerados pela interação não linear entre múltiplas variáveis. Algoritmos baseados em regressão linear simples ou regressão logística terão um desempenho muito pobre aqui e sofrerão instabilidade matemática devido à multicolinearidade. Esse cenário exige duas ações práticas: 1) A consolidação das variáveis redundantes na fase de Engenharia de Recursos (ex: calcular o delta térmico entre processo e ar, ou a potência combinada de torque e RPM) para enxugar o vetor $X$; e 2) O uso definitivo de modelos não lineares avançados (como Random Forest, XGBoost ou Redes Neurais), capazes de encontrar os padrões multidimensionais que causam as avarias.

**Gráfico 4: Fronteiras de Falha Físico-Mecânica (Dispersão Torque vs RPM):** A análise visual do Gráfico 4 revela uma correlação inversamente proporcional e não linear entre o Torque e a Rotação, desenhando a curva clássica de restrição de potência do motor. O padrão isolado mais crítico é que as falhas (marcadores em laranja) não ocorrem de maneira aleatória na nuvem de dados, mas sim agrupadas nas bordas extremas do envelope operacional: na zona de alto estresse e travamento (Torque > 60 Nm acoplado a Rotação < 1500 RPM) e na zona de sobrevelocidade ou ruptura (Rotação > 2250 RPM com Torque < 15 Nm). Esse comportamento confirma que as quebras são delimitadas por fronteiras físicas estritas e não lineares. Para capitalizar sobre esse padrão, a etapa de Engenharia de Recursos (*Feature Engineering*) deverá criar uma nova variável combinada capturando essa dinâmica (ex: `potencia_gerada = torque * rpm`). Adicionalmente, o formato de delimitação nas bordas indica que modelos lineares simples terão dificuldade de classificação, reforçando a escolha por algoritmos baseados em árvores (Random Forest, XGBoost) ou SVMs, que lidam excelentemente com fronteiras de decisão espaciais complexas.