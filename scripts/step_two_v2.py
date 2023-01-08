# python step_two_v2.py --data_frame data.csv.gz --video video.mp4 --fps 25 --chunk 3000
# autopep8 --in-place --aggressive --aggressive script2/main.py
import click
import subprocess
import numpy as np
import pandas as pd
import imageio.v3 as iio


@click.command()
@click.option(
    '--data_frame',
    required=True,
    help='Arquivo CSV com os dados do data frame.'
)
@click.option(
    '--video',
    required=True,
    help='Arquivo de vídeo.'
)
@click.option(
    '--fps',
    default=25,
    help='Numéro de frames por segundo capturados pela câmera'
)
@click.option(
    '--chunk',
    default=3000,
    help='Número de quadros a serem analisados por vez. 3000 a 25 fps é de 2 minutos.'
)
def main(data_frame, video, fps, chunk):
    # Lê o arquivo CSV comprimido e seleciona apenas a coluna 'std'
    data_frame = pd.read_csv(
        f'{data_frame}',
        compression='gzip',
        usecols=['std']
    )

    # Normaliza os valores da coluna 'std' pelo seu máximo
    max_std = data_frame['std'].max()
    data_frame['std'] = data_frame['std'].apply(lambda _: _ / max_std)

    # Armazena os valores da coluna 'std' em uma variável
    std_values = data_frame['std']

    # Parâmetro alpha utilizado no cálculo de tiny
    alpha = 1

    # Calcula o valor médio de std_values mais o desvio padrão multiplicado
    # por alpha
    tiny = np.mean(std_values, axis=0) + alpha * np.std(std_values)

    n = chunk
    # Lista de arquivos gerados
    files = list()
    for chunk in [(i, i + n) for i in range(0, std_values.shape[0], n)]:
        # Desempacota os índices inicial e final do pedaço atual
        start, end = chunk

        # Seleciona os valores de std_values no intervalo start-end
        std_chunk = std_values[start:end]

        # Calcula os coeficientes da função polinomial de grau 7 que melhor se ajusta aos valores de std_chunk
        # O resultado é armazenado nas variáveis coefficients
        coefficients = np.polyfit(
            [_ for _ in range(std_chunk.shape[0])],
            [y[1] for y in std_chunk.items()],
            7,
            full=True
        )[0]

        # Gera a função polinomial com os coeficientes calculados
        f = np.poly1d(coefficients)

        # Variável de contagem utilizada no loop interno
        x = 0

        # Tamanho do intervalo avaliado em cada iteração do loop interno
        interval_size = 5

        # Contador de detecções
        detections = 0

        # Loop interno que verifica se há intervalos de tamanho interval_size
        # em que f(x + i) >= tiny para todo i no intervalo [0, interval_size)
        while x < std_chunk.shape[0]:
            # Se o próximo intervalo exceder a quantidade de quadros
            # disponíveis, saia do loop
            if x + interval_size > n:
                break
            # Se durante interval_size consecutivo, o valor de std for maior
            # que tiny, esse intervalo é um candidato para detecção.
            if all(f(x + i) >= tiny for i in range(interval_size)):
                # Pular para o próximo intervalo
                x += interval_size
                # Incrementar o contador de detecção para o intervalo
                # analisado.
                detections += 1
                continue
            # Pular para o próximo intervalo
            x += 1

        # Se a detecção cumulativa para o intervalo exceder 1 segundo de vídeo
        if detections >= fps:
            # Calcula os índices inicial e final do subclip em segundos
            start_time = start / fps
            end_time = end / fps

            # Gera o comando ffmpeg para gerar o subclip
            cmd = [
                "/usr/bin/ffmpeg", "-y",
                "-ss", "%0.2f" % start_time,
                "-i", f'{video}',
                "-t", "%0.2f" % (end_time - start_time),
                "-map", "0", "-vcodec", "copy", "-acodec", "copy",
                f'./subclip_{int(start_time * fps)}_{int(end_time * fps)}.mp4'
            ]

            files.append(
                f'./subclip_{int(start_time * fps)}_{int(end_time * fps)}.mp4')

            print(" ".join(cmd))

            # Executa o comando

            # Iniciar o processo
            process = subprocess.Popen(cmd)

            # Esperar o processo terminar
            process.wait()

            # Verificar o código de retorno
            if process.returncode != 0:
                # Erro ao executar o comando
                print(
                    "O processo falhou com o código de retorno:",
                    process.returncode)

            # Encerra o processo
            process.terminate()

    # Iterar sobre cada arquivo
    for subclip in [f for f in files if f.endswith('.mp4')]:

        # Abra o vídeo usando iio.imopen()
        video = iio.imopen(uri=subclip, io_mode='r')

        # Inicialize um array para armazenar os frames resultantes
        image = np.zeros(video.properties().shape[1::], dtype=np.uint8)

        # Sobreponha todos os frames um a um
        for frame in video.iter():
            # print(image.shape, frame.shape)
            image = np.maximum(image, frame)

        # Salve a imagem resultante em um arquivo
        print(f'{subclip.replace(".mp4", ".png")}')
        iio.imwrite(f'{subclip.replace(".mp4", ".png")}', image)

        # Feche o vídeo
        video.close()


if __name__ == '__main__':
    main()
