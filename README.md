# Guia de Execução do Notebook

Este guia provê as instruções necessárias para configurar o ambiente e executar o notebook de Machine Learning para a detecção de Fake News, utilizando diversos modelos e acompanhamento via MLflow.

## 1. Pré-requisitos
Para executar o notebook, você precisa ter o Python instalado, preferencialmente nas versões 3.8 a 3.12, conforme verificado no script de configuração.


### 1.1. Arquivos Essenciais (Download Manual)
O notebook depende de três arquivos externos que devem ser baixados e colocados na raiz do seu projeto (onde o notebook está localizado):

Dataset Principal:

- Nome esperado no arquivo ZIP: Fake News Dataset.zip (ou nomes alternativos como fake_news_dataset.zip que o script procura ).

- Onde colocar: Na pasta raiz do projeto, na subpasta data, ou na sua pasta Downloads ou Desktop .

Observação: O script de carregamento tentará encontrar este arquivo em várias localizações comuns.

Embeddings GloVe Pré-treinados:

Estes arquivos fazem parte do pacote GloVe, hospedado pela Stanford NLP.

Link de Download do ZIP: https://nlp.stanford.edu/data/glove.6B.zip 

Arquivos necessários:

- glove.6B.100d.txt (100 dimensões) 
- glove.6B.300d.txt (300 dimensões) 

Onde colocar: Descompacte o arquivo e copie os dois arquivos .txt acima para a raiz do seu projeto.

# 2. Configuração do Ambiente e Instalação de Pacotes
O notebook contém uma seção de configuração inicial (Seção 1) que instala todos os pacotes necessários (PyTorch, Transformers, Scikit-learn, MLflow, etc.) .

## 2.1. Requisitos para Aceleração por GPU (Opcional)
Se você tiver uma GPU NVIDIA, o script tentará configurar e instalar a versão compatível do PyTorch com CUDA .

Para garantir a compatibilidade, certifique-se de que o driver NVIDIA e o kit de ferramentas CUDA (se já instalados) estejam atualizados. O script tentará detectar a melhor versão (e.g., 11.8 ou 12.1) .

## 2.2. Processo de Instalação
Abra o seu ambiente de desenvolvimento (Jupyter, VS Code, etc.) e execute a primeira célula do notebook (Seção 1) que contém as seguintes funções e a chamada principal main():

set_encoding(): Configura a codificação UTF-8.

check_python_version(): Verifica a compatibilidade do Python.

detect_gpu_safe() / get_cuda_version_safe(): Detecta a GPU e a versão do CUDA.

install_pytorch_gpu_safe() / install_pytorch_cpu_safe(): Instala o PyTorch (com ou sem GPU).

install_compatible_packages_safe(): Instala todos os pacotes de ML, incluindo transformers, mlflow, xgboost, scikit-learn, optuna, etc., nas versões fixadas para garantir a compatibilidade .

Aguarde a conclusão: A instalação de todos os pacotes, especialmente as dependências do PyTorch e dos Transformers, pode levar alguns minutos.

# 3. Acompanhamento com MLflow
O notebook está configurado para registrar o ciclo de vida dos experimentos (LODO, otimização, modelos) usando o MLflow.

## 3.1. Configuração do Tracking Local
A célula 1.3 configura o MLflow para salvar os dados de rastreamento localmente no diretório mlruns:
```
# Diretório de salvamento (na raiz do projeto)
PROJECT_ROOT = pathlib.Path.cwd()
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
```

- Nome do Experimento: Todos os seus runs serão agrupados sob o experimento principal chamado fake-news-multimodel.

## 3.2. Visualizando a Interface do MLflow
Para acompanhar as métricas, parâmetros, modelos salvos e artefatos, abra uma nova janela do terminal (na raiz do seu projeto) e execute o seguinte comando:
```
mlflow ui
```
- O terminal indicará a URL para acessar a interface web (geralmente http://localhost:5000 ou similar).

- O que você verá:

-- Runs Raiz: LODO-classic (modelos tradicionais) , LODO-BERT (modelos baseados em Transformers), e runs de otimização (Optuna).

-- Runs Filhas (Nested Runs): Dentro do LODO-classic e LODO-BERT, você verá runs para cada holdout (dataset de teste, e.g., holdout=ISOT Fake News Dataset), contendo as métricas de generalização (F1-Macro, AUPRC) e os modelos salvos.

# 4. Execução do Fluxo de Trabalho (Workflow)
Após a instalação dos pacotes, o fluxo de trabalho segue as seguintes seções do notebook:

Carregando os dados (Seção 2): Processa o Fake News Dataset.zip, combina os 8 datasets internos (incluindo ISOT, Kaggle e WELFake), e salva o cache em fake_news_processed.csv .

Limpeza dos Dados (Seção 3): Aplica a limpeza, incluindo a mascaramento de entidades nomeadas (NER) via spaCy e remoção de stopwords .

Carregar Embeddings (Seção 4): Carrega os arquivos GloVe 100d e 300d que você baixou na raiz do projeto.

LODO Experiment (Seção 5): Executa a validação Leave-One-Dataset-Out (LODO) para todos os modelos, registrando cada fold no MLflow.

Recomendação: Execute o notebook célula por célula, revisando as saídas e, se necessário, o progresso no MLflow.
