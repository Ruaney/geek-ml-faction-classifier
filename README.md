# Geek ML Faction Classifier

Um projeto pequeno e didático de machine learning: um classificador prevê a facção de um personagem de uma galáxia fictícia a partir de atributos numéricos.

As facções são originais e o conjunto de dados foi criado apenas para demonstração. O objetivo é mostrar um fluxo completo e fácil de entender:

1. carregar dados CSV;
2. separar treino e teste;
3. treinar uma regressão logística;
4. avaliar acurácia e relatório por classe;
5. salvar o modelo e fazer previsões pela linha de comando.

## Classes

- `Sentinel`: defensores equilibrados, com alta diplomacia e disciplina.
- `Void Raider`: especialistas em combate e invasão.
- `Circuit Sage`: personagens focados em tecnologia e cibernética.

## Como executar

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
```

No Windows:

```bash
.venv\\Scripts\\activate
```

No macOS/Linux:

```bash
source .venv/bin/activate
```

Instale as dependências e treine o modelo:

```bash
pip install -r requirements.txt
python train.py
```

Depois, experimente uma previsão:

```bash
python predict.py --combat 8 --diplomacy 3 --technology 6 --cybernetics 4
```

## Estrutura

```text
.
├── data/characters.csv   # dataset sintético
├── train.py              # treinamento e avaliação
├── predict.py            # previsão via terminal
├── requirements.txt
└── README.md
```

## Observação

O dataset é propositalmente pequeno para manter o experimento rápido e legível. Para evoluir o projeto, adicione exemplos ao CSV, compare outros modelos e use validação cruzada.

## Versão Hugging Face: Geek Genre Classifier

O repositório também inclui uma trilha de fine-tuning com `distilbert-base-uncased`. Ela classifica descrições curtas em quatro gêneros geek:

- `cyberpunk`
- `high_fantasy`
- `space_opera`
- `superhero`

Os textos em `data/geek_genres.csv` são exemplos originais criados para fins educacionais. Isso torna o modelo apropriado como demonstração de portfólio, não como um classificador de produção ou benchmark de gênero literário.

### Treinar o modelo para o Hub

```bash
pip install -r requirements-hf.txt
python train_hf.py --epochs 8
```

Após o treino, siga [HUGGINGFACE.md](HUGGINGFACE.md) para autenticar, publicar o modelo e criar a Space em Gradio.

## Licença

MIT
