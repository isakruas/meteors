#!/bin/bash

# Script para criar sequências de imagens PNG a partir de arquivos de vídeo WebM

# Itera através do diretório de vídeo
for video in $(ls -p ./webcam/ | grep -v /); do
    # Verifica se o diretório de saída para o vídeo atual já existe
    if [ ! -d "./imagens/$(basename "$video" .webm)" ]; then
        # Cria o diretório de saída para o vídeo atual
        mkdir -p "./imagens/$(basename "$video" .webm)"
        # Cria uma sequência de imagens PNG a partir do arquivo de vídeo atual,
        # usando um intervalo de 0 a 30 frames e um tamanho máximo de 299 pixels,
        # e salva as imagens geradas no diretório de saída correspondente ao vídeo atual
        echo "Processando $video ..."
        ./apps/png-seq-creator/png-seq-creator 0 30 299 ./webcam/$video ./imagens/$(basename "$video" .webm)/
    fi
done
