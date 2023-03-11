#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    // 1. Verifica se a quantidade de argumentos é válida
    if (argc != 4)
    {
        cerr << "Usage: " << argv[0] << " <crop_size> <image_path> <save_files_path>" << endl;
        return 1;
    }

    // 2. Converte o tamanho do recorte para inteiro
    int crop_size = std::stoi(argv[1]);

    // 3. Carrega a imagem em escala de cinza
    Mat img = imread(argv[2], IMREAD_GRAYSCALE);

    // 4. Salva o caminho para onde serão salvos os recortes
    std::string save_files_path = argv[3];

    // 5. Verifica se a imagem foi carregada corretamente
    if (img.empty())
    {
        cerr << "Could not read image: " << argv[2] << endl;
        return 1;
    }

    // 6. Define o ponto inicial de onde será feito o recorte
    int x_start = 0;
    int y_start = 0;

    // 7. Percorre a imagem fazendo recortes de tamanho "crop_size" em cada direção
    for (int y = y_start; y < img.rows - crop_size; y += crop_size)
    {
        for (int x = x_start; x < img.cols - crop_size; x += crop_size)
        {
            // 8. Cria o retângulo que representa o recorte
            Rect roi(x, y, crop_size, crop_size);

            // 9. Faz o recorte da imagem usando o retângulo
            Mat crop = img(roi);

            // 10. Define o nome do arquivo que será salvo, usando a posição do recorte na imagem
            std::string filename = std::to_string(x) + "_" + std::to_string(y) + ".png";

            // 11. Salva o recorte em disco, no caminho especificado
            imwrite(save_files_path + filename, crop);
        }
    }

    return 0;
}
