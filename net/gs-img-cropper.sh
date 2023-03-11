#!/bin/bash

# Script para processar imagens em lotes

# Itera através do diretório de imagens e seus subdiretórios
for video in $(ls -p ./imagens/ | grep / | sed 's|/||g'); do
    # Itera através das imagens do subdiretório atual
    for img in $(ls -p ./imagens/$video/ | grep -v /); do
        # Verifica se o diretório de saída para a imagem atual já existe
        if [ ! -d "./imagens/$video/datasets/$(basename "$img" .png)" ]; then
            # Cria o diretório de saída para a imagem atual
            mkdir -p "./imagens/$video/datasets/$(basename "$img" .png)"
        fi
        # Processa a imagem atual usando o aplicativo "gs-img-cropper"
        # com um valor de 27 como argumento e salva a imagem processada no diretório de saída
        ./apps/gs-img-cropper/gs-img-cropper 27 ./imagens/$video/$img ./imagens/$video/datasets/$(basename "$img" .png)/
    done
done
