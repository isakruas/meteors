# autopep8 --in-place --aggressive --aggressive scripts/step_one.py
from concurrent.futures.process import ProcessPoolExecutor as Executor
import sys
import multiprocessing
import pandas as pd
import numpy as np
# Importa o módulo convolve da biblioteca scipy, responsável por realizar
# a convolução de imagens
from scipy.signal import convolve
# Importa o módulo iio da biblioteca imageio, responsável por realizar a
# leitura de imagens e vídeos
import imageio.v3 as iio
import gc


def preprocessing(figure):
    # Verifica se a imagem possui 3 canais de cor (RGB)
    # Caso contrário, assume que a imagem é em tons de cinza e não precisa ser
    # alterada
    try:
        # Seleciona apenas um canal de cor da imagem, calculando a soma dos
        # três canais
        figure = figure[:, :, 0] + figure[:, :, 1] + figure[:, :, 2]
    except IndexError:
        # A imagem é em tons de cinza, então não há necessidade de realizar
        # nenhuma alteração
        pass

    # Calcula o desvio padrão de todos os píxeis da imagem
    std = np.std(figure.reshape((figure.shape[0] * figure.shape[1])))

    # Cria um filtro de normalização para a imagem
    # O filtro é uma matriz 7x7 com todos os elementos iguais a 1,0/std
    normalization_filter = np.zeros((7, 7))
    normalization_filter[:] = 1.0 / std
    # Aplica o filtro de normalização na imagem, utilizando a função convolve
    figure = convolve(figure, normalization_filter, mode='valid')

    # Cria um filtro de Laplace
    # O filtro é uma matriz 3x3 com todos os elementos iguais a -1, exceto o
    # elemento central que é 8
    laplace_filter = np.ones((3, 3)) * -1
    laplace_filter[1, 1] = 8
    # Aplica o filtro de Laplace na imagem, utilizando a função convolve
    figure = convolve(figure, laplace_filter, mode='valid')

    # Remove informações não relevantes da imagem
    # Utiliza operações matemáticas para remover os píxeis com valores muito
    # próximos de zero
    figure = figure - (np.abs(figure) * (-0.99))
    figure = figure * -(figure - np.abs(figure))

    # Cria um filtro personalizado para destacar as estrelas
    # O filtro é uma matriz 7x7, criada manualmente com valores pré-definidos
    ruas_filter = np.flip(np.array([
        [-99, -90, -93, -118, -92, -90, -99],
        [-105, -79, -98, -170, -98, -79, -105],
        [-112, -99, -122, -122, -122, -99, -112],
        [-118, -170, -122, -255, -122, -170, -118],
        [-112, -99, -122, -122, -122, -99, -112],
        [-105, -79, -98, -170, -98, -79, -105],
        [-99, -90, -93, -118, -92, -90, -99],
    ]))
    # Aplica o filtro personalizado na imagem, utilizando a função convolve
    return convolve(figure, ruas_filter, mode='valid')


def processing(frame):
    # Calcula o desvio padrão do frame processado
    # O método .values retorna os valores do DataFrame como um array numpy
    # O método .flatten() transforma o array num vetor unidimensional
    # O método .std() calcula o desvio padrão do vetor
    std = pd.DataFrame(preprocessing(frame)).values.flatten().std()
    return std


# Cria um DataFrame vazio para armazenar os valores de desvio padrão
df = pd.DataFrame({'std': []})

# Cria um Lock
lock = multiprocessing.Lock()


def callback(future):
    global lock
    # Trava o Lock
    lock.acquire()
    global df
    # Adiciona o valor de desvio padrão ao DataFrame
    df = pd.concat(
        [df, pd.DataFrame({'std': [future.result()]})], ignore_index=True)
    print(df.shape[0], end='\r', flush=True)
    # Executa a coleta de lixo
    gc.collect()
    # Destrava o Lock
    lock.release()


def main(uri):
    # Abre o vídeo para leitura
    # O parâmetro io_mode='r' indica que o vídeo será apenas lido, sem escrita
    video = iio.imopen(uri=uri, io_mode='r')
    # Cria um objeto responsável por gerir a execução de tarefas
    # em paralelo. O objeto é inicializado com o número máximo de trabalhadores
    # igual a 10 vezes o número de núcleos de CPU disponíveis. Isso significa
    # que o pré-processamento de imagens será realizado em paralelo por um número
    # de processos igual a 10 vezes o número de núcleos de CPU disponíveis.
    with Executor(max_workers=multiprocessing.cpu_count()) as executor:
        # Itera sobre cada frame do vídeo
        for frame in video.iter():
            # Envia uma tarefa para ser executada de maneira assíncrona
            future = executor.submit(processing, frame)
            # Adiciona a função de callback ao objeto Future
            future.add_done_callback(callback)


if __name__ == "__main__":
    # Verifica se foi passado um argumento
    if len(sys.argv) < 2:
        print("Error: você deve informar o nome do arquivo de vídeo como argumento.")
        sys.exit(1)

    # Obtém o nome do arquivo de vídeo
    video_file = sys.argv[1]

    # Inicia o processamento de vídeo
    main(video_file)

    # Salva o DataFrame com os valores de desvio padrão num arquivo .csv.gz
    # O parâmetro compression='gzip' indica que o arquivo será compactado
    # utilizando o algoritmo gzip
    df.to_csv('data.csv.gz', compression='gzip')
