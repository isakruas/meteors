# Descrição do Código

Este código realiza o recorte de imagens em escala de cinza em várias direções, de acordo com um tamanho especificado pelo usuário. O usuário deve informar o tamanho do recorte, o caminho da imagem a ser recortada e o caminho onde serão salvos os arquivos de recorte.

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
./gs-img-cropper <crop_size> <image_path> <save_files_path>
```

Os argumentos têm o seguinte significado:

- `<crop_size>`: tamanho do recorte em pixels.
- `<image_path>`: caminho da imagem a ser recortada.
- `<save_files_path>`: caminho para onde serão salvos os recortes.

## Fluxo do programa

O programa segue o seguinte fluxo:

1. Verifica se a quantidade de argumentos é válida.
2. Converte o tamanho do recorte para inteiro.
3. Carrega a imagem em escala de cinza.
4. Salva o caminho para onde serão salvos os recortes.
5. Verifica se a imagem foi carregada corretamente.
6. Define o ponto inicial de onde será feito o recorte.
7. Percorre a imagem fazendo recortes de tamanho "crop_size" em cada direção.
8. Cria o retângulo que representa o recorte.
9. Faz o recorte da imagem usando o retângulo.
10. Define o nome do arquivo que será salvo, usando a posição do recorte na imagem.
11. Salva o recorte em disco, no caminho especificado.
