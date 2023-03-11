# Descrição do Código

Este código processa um arquivo de vídeo e cria uma sequência de imagens PNG sobrepostas em intervalos regulares. O resultado final é uma série de imagens em escala de cinza redimensionadas.

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
./png-seq-creator <start_interval> <duration_in_seconds> <frame_size> <video_file_path> <save_files_path>
```


Os argumentos têm o seguinte significado:

- `<start_interval>`: o número do intervalo a partir do qual o processamento deve começar
- `<duration_in_seconds>`: a duração de cada intervalo em segundos
- `<frame_size>`: o tamanho em pixels para o qual a imagem será redimensionada
- `<video_file_path>`: o caminho completo para o arquivo de vídeo a ser processado
- `<save_files_path>`: o caminho completo para a pasta onde as imagens processadas serão salvas

## Fluxo do programa

O programa segue o seguinte fluxo:

1. Verificar se o número correto de argumentos foi passado
2. Extrair os argumentos de linha de comando
3. Carregar o arquivo de vídeo
4. Verificar se o arquivo foi carregado corretamente
5. Definir a taxa de quadros do vídeo
6. Definir o número total de quadros no vídeo
7. Definir o número de intervalos que serão criados
8. Loop para processar cada intervalo
9. Definir o tempo de início e fim do intervalo
10. Posicionar o vídeo no tempo de início do intervalo
11. Capturar o primeiro quadro do intervalo
12. Loop para processar cada quadro do intervalo
13. Posicionar o vídeo no tempo do próximo quadro
14. Capturar o próximo quadro
15. Sobrepor os quadros mantendo os pixels mais claros
16. Normalizar a imagem dividindo cada valor de pixel pelo valor máximo da imagem e multiplicando pelo valor máximo de intensidade possível
17. Converter o frame para escala de cinzas
18. Redimensionar a imagem para frame_sizexframe_size pixels
19. Salvar o quadro final do intervalo
20. Liberar recursos e encerrar o programa
