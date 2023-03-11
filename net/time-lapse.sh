#!/bin/bash

# Script para criar time-lapse das imagens

# Itera através do diretório de imagens e seus subdiretórios
for path in $(ls -p ./imagens/ | grep / | sed 's|/||g'); do
    if [ ! -d "./imagens/$path/time-lapse/" ]; then
        # Cria o diretório de saída para a imagem atual
        mkdir -p "./imagens/$path/time-lapse/"
    fi
    # Executa o aplicativo de criação de time-lapse com os parâmetros especificados
    ./apps/time-lapse/time-lapse 30 672 897 ./imagens/$path/ ./imagens/$path/time-lapse/
done
