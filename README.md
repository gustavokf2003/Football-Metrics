# **Football Game Processor**

Este projeto foi desenvolvido para processar jogos de futebol, analisando métricas durante uma partida. Ele utiliza técnicas de processamento de imagem e detecção de objetos para rastrear os jogadores no campo.

## **Funcionalidades**
- Detecção de jogadores e bola
- Detecção de pontos no campo
- Rastreamento dos jogadores
- Cálculo do centróide, da profundidade e da amplitude da equipe
- Calcula da posse de bola da equipe
- Cálculo da velocidade de cada jogador com base nas mudanças de posição.
- Cálculo da entropia de cada jogador
- Contagem de passes de cada jogador
- Visualização da posição dos jogadores e da bola em um mapa 2D.

## **Instalação**
```bash
pip install -r requirements.txt
```

### Uso

Modifique as variáveis source_video_path e target_video_path do arquivo main.py com a localização do video que você deseja processar e localização do video de saída, respectivamente.

Depois basta rodar esse script.

```bash
python src/main.py
```
