# autopep8 --in-place --aggressive --aggressive scripts/step_two.py
# Importa os módulos os e matplotlib.pyplot
import os
import matplotlib.pyplot as plt
# Importa o módulo Pandas
import pandas as pd
# Importa o módulo numpy
import numpy as np

# Carrega o conjunto de dados para um DataFrame do pandas
df = pd.read_csv('data.csv.gz', compression='gzip', usecols=['std'])

# Dividimos os valores da coluna 'std' por 3 para normalizá-los, pois eles são
# cumulativos dos três canais de cores: vermelho, verde e azul
# utilizados para calcular o desvio padrão
df['std'] /= 3


# Define o valor de alpha
alpha = 3.7

# Calcula o valor 'tiny' como a média dos dados mais o alpha vezes o
# desvio padrão
tiny = np.mean(df, axis=0) + alpha * np.std(df)

# Divide os dados em trechos de 3000 quadros
for chunk in [(i, i + 3000) for i in range(0, df.shape[0], 3000)]:
    # Define os tempos inicial e final em quadros
    t1 = chunk[0]
    t2 = chunk[1]

    # Seleciona os dados do trecho atual
    std = df[t1:t2]

    # Verifica se o valor máximo do trecho atual é maior ou igual ao valor
    # 'tiny'
    if std.max(axis=0)[0] >= tiny[0]:
        # Cria um gráfico do trecho atual
        plt.figure()

        # Obtém o objeto de eixos do gráfico
        ax = plt.gca()

        # Desenha a série de dados no gráfico
        plt.plot(std)

        # Adiciona uma linha horizontal vermelha na posição y=30
        ax.axhline(y=tiny[0], color='r')

        # Adiciona labels nos eixos do gráfico
        plt.xlabel(f'TIME {t1}:{t2}')
        plt.ylabel('STD')

        plt.savefig(f'./subclip_{t1}_{t2}.png')
        # Exibe o gráfico
        # plt.show()
        plt.close()

        # Define os tempos inicial e final em segundos
        t1 = t1 / 25
        t2 = t2 / 25

        # Cria a lista de comandos para o ffmpeg
        cmd = [
            "/usr/bin/ffmpeg", "-y",  # opções do ffmpeg
            "-ss", "%0.2f" % t1,  # tempo inicial
            "-i", f'./video.mp4',  # arquivo de entrada
            "-t", "%0.2f" % (t2 - t1),  # duração do trecho a ser cortado
            "-map", "0", "-vcodec", "copy", "-acodec", "copy",  # opções de codificação
                   # arquivo de saída
                   f'./subclip_%s_%s.mp4' % (int(t1 * 25), int(t2 * 25))
        ]
        print(" ".join(cmd))
        # Executa o comando
        os.system(" ".join(cmd))
