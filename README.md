# Pipeline de Treinamento e Avaliação (YOLOv26 + K-Fold)

Este projeto executa uma pipeline completa para:
1. Gerar dobras de validação cruzada (K-Fold) no formato COCO Json.
2. Treinar múltiplos modelos YOLOv26 em cada dobra.
3. Consolidar métricas e gerar gráficos comparativos.

## Estrutura esperada

```text
dataset/
  train/
    _annotations.coco.json
    <imagens>
src/
  data/
    generate_folds.py
  train/
    train_kfold_models.py
  eval/
    analyze_results.py
```

## Pré-requisitos

- Python 3.10+
- Dependências do arquivo `environment/env_compara_yolo_torch.yml`
- CUDA (opcional, para GPU)

### Criar ambiente Conda

```bash
conda env create -f environment/env_compara_yolo_torch.yml
conda activate compara_yolo_torch
```

Obs. Se desejar criar um ambiente com outro nome, edite o arquivo `environment/env_compara_yolo_torch.yml` e substitua o nome do seu ambiente na primeira linha.

## 1) Preparar dataset base

O dataset de entrada deve estar em COCO JSON:
- `dataset/train/_annotations.coco.json`
- `dataset/train/<imagens>`

Nesse formato, o script:
- lê `images`, `annotations` e `categories` do COCO;
- converte as caixas para labels YOLO (`.txt`) por imagem;
- gera os folds já prontos para treino com Ultralytics.

## 2) Gerar folds

Script: `src/data/generate_folds.py`

Esse script:
- lê as imagens e anotações em COCO JSON do dataset base;
- cria `folds/fold_1`, `folds/fold_2`, ... `folds/fold_K`;
- gera `data.yaml` de cada dobra com `nc` e `names`.


### Execução padrão

```bash
python src/data/generate_folds.py
```

### Configuração no código

Para definir o número de folds, abra o arquivo `src/data/generate_folds.py` e modifique as linhas abaixo:
- `K`
- `RANDOM_STATE`

## 3) Treinar modelos em K-Fold

Script: `src/train/train_kfold_models.py`

Esse script:
- percorre todas as dobras em `folds/`;
- treina cada modelo da lista `MODEL_NAMES`;
- salva saídas em `results/fold_<n>_<modelo>/`.

### Configuração no código

No topo do arquivo, ajuste:
- `MODEL_NAMES`
- `TRAIN_CONFIG` (`epochs`, `batch`, `imgsz`, etc.)
- `DEVICE`
- `CLEAN_RESULTS_DIR`

### Execução

```bash
python src/train/train_kfold_models.py
```

### Saída esperada

Para cada combinação dobra+modelo:
- `results/fold_<n>_<modelo>/results.csv`
- `results/fold_<n>_<modelo>/weights/best.pt`
- `results/fold_<n>_<modelo>/weights/last.pt`
- gráficos padrão do Ultralytics (`results.png`, curvas, matriz de confusão, etc.)

## 4) Analisar resultados

Script: `src/eval/analyze_results.py`

Esse script:
- encontra execuções com padrão `fold_<n>_<modelo>`;
- consolida métricas por dobra e por modelo;
- gera tabelas e gráficos em `results/analysis`.

### Execução

```bash
python src/eval/analyze_results.py
```

### Tabelas geradas

- `results/analysis/per_fold_metrics.csv`
- `results/analysis/model_summary.csv`
- `results/analysis/model_ranking.csv`
- `results/analysis/stability_table.csv`

### Gráficos gerados

- Boxplots por métrica (`precision`, `recall`, `map50`, `map50_95`)
- Barras com `mean ± std` por métrica
- Heatmap `fold x modelo` para `mAP50-95`
- Scatter `precision vs recall`
- Ranking horizontal por `mAP50-95`
- Curvas por época (`train_box_loss`, `val_box_loss`, `map50_95`)

## Ordem dos modelos na análise

A análise exibe os modelos na ordem de tamanho:
- `yolo26n`
- `yolo26s`
- `yolo26m`
- `yolo26l`
- `yolo26x`

## Pipeline resumida (comandos)

```bash
# 1) gerar folds
python src/data/generate_folds.py

# 2) treinar todos os modelos em todas as dobras
python src/train/train_kfold_models.py

# 3) consolidar e plotar resultados
python src/eval/analyze_results.py
```

## Dicas de troubleshooting

- Erro de `results.csv` ausente:
  Verifique se o treino terminou para todas as pastas em `results/`.

- Erro de labels faltando:
  Verifique se `dataset/train/_annotations.coco.json` contém anotações válidas e se as imagens referenciadas em `images[].file_name` existem dentro de `dataset/train/`.

- Erro de classe (`nc` vs `names`):
  Verifique se `categories` no COCO está consistente (`id` e `name`).

- Sem GPU:
  O treino usa CPU automaticamente quando CUDA não está disponível.
