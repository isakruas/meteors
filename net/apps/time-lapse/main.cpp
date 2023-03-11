#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <string.h>
#include <algorithm>

using namespace cv;
using namespace std;

// Função para comparar os nomes de arquivos
bool compare_file_names(const string &a, const string &b)
{
    // Extrai os números do nome dos arquivos
    int nun_a, nun_b;
    try
    {
        string a_num_str = a.substr(a.find_last_of('/') + 1, a.find_last_of('_') - a.find_last_of('/') - 1);
        string b_num_str = b.substr(b.find_last_of('/') + 1, b.find_last_of('_') - b.find_last_of('/') - 1);
        nun_a = stoi(a_num_str);
        nun_b = stoi(b_num_str);
    }
    catch (const std::invalid_argument &e)
    {
        // Se um dos nomes não estiver no formato esperado, retorna true para que o arquivo fique no final da lista
        return true;
    }

    // Compara os números dos arquivos
    return nun_a < nun_b;
}

// Função principal
int main(int argc, char **argv)
{
    // 1. Verificar se o número correto de argumentos foi passado
    if (argc != 6)
    {
        std::cout << "Usage: " << argv[0] << " <fps> <frame_size_w> <frame_size_h> <dir_path_imagens> <save_video_path>" << std::endl;
        return -1;
    }

    // Define as configurações do vídeo
    int fps = std::stoi(argv[1]); // Frames por segundo do vídeo
    int frame_size_w = std::stoi(argv[2]);
    int frame_size_h = std::stoi(argv[3]);
    Size frame_size(frame_size_w, frame_size_h); // Tamanho dos frames do vídeo
    Scalar background_color(0, 0, 0);            // Cor de fundo preta
    string video_name = "video.avi";             // Nome do arquivo de vídeo

    // Extrair os argumentos de linha de comando
    std::string dir_path_imagens = argv[4];
    std::string save_video_path = argv[5];

    // Cria uma lista vazia de imagens
    vector<string> image_list;

    // Abre o diretório
    DIR *dir = opendir(dir_path_imagens.c_str());

    // Verifica se o diretório foi aberto corretamente
    if (dir)
    {
        struct dirent *entry;

        // Itera sobre os arquivos no diretório
        while ((entry = readdir(dir)) != NULL)
        {
            string file_name = entry->d_name;

            // Verifica se a entrada é um arquivo com extensão .jpg ou .png
            if (file_name.find(".jpg") != string::npos || file_name.find(".png") != string::npos)
            {
                // Adiciona o caminho completo do arquivo à lista de imagens
                string file_path = dir_path_imagens + file_name;
                image_list.push_back(file_path);
            }
        }

        // Fecha o diretório
        closedir(dir);
    }
    else
    {
        // Se o diretório não puder ser aberto, exibe uma mensagem de erro e sai do programa
        cout << "Erro ao abrir o diretório " << dir_path_imagens << endl;
        return 1;
    }

    // Ordena a lista de imagens pelo nome dos arquivos usando a função compare_file_names
    sort(image_list.begin(), image_list.end(), compare_file_names);

    // Cria um objeto VideoWriter para gravar o vídeo
    VideoWriter video(save_video_path + video_name, VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frame_size);
    for (const auto &imagePath : image_list)
    {
        // Carrega a imagem original
        Mat image = imread(imagePath);

        // Cria um frame com a cor de fundo preta
        Mat frame(frame_size, CV_8UC3, background_color);

        // Calcula as coordenadas para centralizar a imagem no frame
        int x = (frame_size.width - image.cols) / 2;
        int y = (frame_size.height - image.rows) / 2;

        // Copia a imagem redimensionada para o frame centralizado
        Mat roi = frame(Rect(x, y, image.cols, image.rows));
        image.copyTo(roi);

        // Grava o frame no vídeo
        video.write(frame);
    }

    // Libera o objeto VideoWriter
    video.release();

    return 0;
}
