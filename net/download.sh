#!/bin/bash

# Carregar as variáveis de ambiente do arquivo de configuração
source config.sh

# Verifica se um arquivo já está baixado na pasta especificada e, caso não esteja, faz o download a partir de um servidor remoto
function download_if_not_exists {
    # Verificar se o arquivo já existe na pasta local
    if [ -f "$2/$1" ]; then
        echo "O arquivo $1 já foi baixado."
    else
        # Fazer o download a partir do servidor remoto 
        echo "Baixando $1 ..."
        sshpass -p "$METEORS_REMOTE_PASSWORD" scp $METEORS_REMOTE_USER@$METEORS_REMOTE_SERVER:"$3/$1" "$2/"
        echo "O arquivo $1 foi baixado com sucesso."
    fi
}

# Conectar via SSH a METEORS_REMOTE_USER@METEORS_REMOTE_SERVER e obter a lista de arquivos da pasta METEORS_REMOTE_PATH
files=$(sshpass -p "$METEORS_REMOTE_PASSWORD" ssh "$METEORS_REMOTE_USER"@"$METEORS_REMOTE_SERVER" ls "$METEORS_REMOTE_PATH")

# Iterar sobre a lista de arquivos e chamar a função download_if_not_exists para cada arquivo
for file in $files; do
    download_if_not_exists "$file" "$METEORS_LOCAL_PATH" "$METEORS_REMOTE_PATH"
done
