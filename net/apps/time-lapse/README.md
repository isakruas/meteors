# Descrição do Código

Este programa é capaz de criar um vídeo a partir de um conjunto de imagens em sequência. Ele lê as imagens de um diretório especificado pelo usuário e as organiza em ordem alfabética de nome de arquivo. Em seguida, cria um vídeo a partir das imagens em sequência e o salva em um arquivo especificado pelo usuário.

## Pré-requisitos

Para executar este código, você precisará ter instalado:

- OpenCV 2 ou superior
- C++ 11 ou superior

## Como usar

Para usar o código, siga os passos abaixo:

1. Compile o código usando o seguinte comando:

```sh 
make
```

2. Execute o executável com os seguintes argumentos:
```sh 
./time-lapse <fps> <frame_size_w> <frame_size_h> <dir_path_imagens> <save_video_path>
```


Os argumentos têm o seguinte significado:

- `fps`: é a taxa de quadros por segundo do vídeo
- `frame_size_w`: é a largura do quadro do vídeo
- `frame_size_h`: é a altura do quadro do vídeo
- `dir_path_imagens`: é o caminho para o diretório contendo as imagens
- `save_video_path`: é o caminho para o arquivo de vídeo que será criado


## Funcionamento

O programa começa verificando se o número correto de argumentos foi passado na linha de comando. Se não, exibe uma mensagem de uso e sai. Se sim, define as configurações do vídeo com base nos argumentos passados. Ele então extrai o caminho para o diretório de imagens e o caminho para o arquivo de vídeo do vetor de argumentos.

Em seguida, cria uma lista vazia de imagens e abre o diretório especificado pelo usuário. Se o diretório não puder ser aberto, o programa exibe uma mensagem de erro e sai. Caso contrário, o programa itera sobre os arquivos no diretório e adiciona à lista de imagens os arquivos com extensão .jpg ou .png. A lista de imagens é então ordenada pelo nome dos arquivos, usando uma função auxiliar que extrai os números dos nomes dos arquivos para compará-los.

Por fim, o programa cria um objeto VideoWriter e itera sobre a lista de imagens. Para cada imagem, o programa carrega a imagem original, cria um quadro preto com o tamanho especificado e cola a imagem centralizada nesse quadro. O quadro resultante é então gravado no arquivo de vídeo. Quando todas as imagens foram processadas, o programa libera o objeto VideoWriter e termina sua execução.
