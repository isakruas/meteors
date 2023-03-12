#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace cuda;

// ./cuda-png-seq-creator 512 30 299 ./2023-03-06-193527.webm ./imagens
int main(int argc, char **argv)
{
    // 1. Verificar se o número correto de argumentos foi passado
    if (argc != 6)
    {
        std::cout << "Usage: " << argv[0] << " <start_interval> <duration_in_seconds> <frame_size> <video_file_path> <save_files_path>" << std::endl;
        return -1;
    }

    // 2. Extrair os argumentos de linha de comando
    int start_interval = std::stoi(argv[1]);
    int duration_in_seconds = std::stoi(argv[2]);
    int frame_size = std::stoi(argv[3]);
    std::string video_file_path = argv[4];
    std::string save_files_path = argv[5];

    // 3. Carregar o arquivo de vídeo
    VideoCapture cap(video_file_path);

    // 4. Verificar se o arquivo foi carregado corretamente
    if (!cap.isOpened())
    {
        std::cout << "Erro ao abrir arquivo de vídeo" << std::endl;
        return -1;
    }

    // 5. Definir a taxa de quadros do vídeo
    double fps = cap.get(CAP_PROP_FPS);

    // 6. Definir o número total de quadros no vídeo
    int total_frames = cap.get(CAP_PROP_FRAME_COUNT);

    // 7. Definir o número de intervalos que serão criados
    int num_intervals = total_frames / (fps * duration_in_seconds);

    // 8. Loop para processar cada intervalo
    for (int i = start_interval; i < num_intervals; i++)
    {

        // 9. Definir o tempo de início e fim do intervalo
        double start_time = i * duration_in_seconds;
        double end_time = (i + 1) * duration_in_seconds;

        // 10. Posicionar o vídeo no tempo de início do intervalo
        cap.set(CAP_PROP_POS_MSEC, start_time * 1000);

        // 11. Capturar o primeiro quadro do intervalo
        Mat frame, next_frame;
        cap.read(frame);
        cuda::GpuMat gpu_frame, gpu_gray_frame, gpu_next_frame;

        // Verificar se chegou ao final do vídeo
        if (frame.empty())
        {
            break;
        }

        // Converter o quadro para o formato CUDA
        gpu_frame.upload(frame);

        // 12. Loop para processar cada quadro do intervalo
        for (int j = 1; j < (int)(fps * duration_in_seconds); j++)
        {
            // 13. Posicionar o vídeo no tempo do próximo quadro
            cap.set(CAP_PROP_POS_MSEC, (start_time + j / fps) * 1000);

            // 14. Capturar o próximo quadro
            cap.read(next_frame);

            // Converter o quadro para o formato CUDA
            gpu_next_frame.upload(next_frame);

            // 15. Sobrepor os quadros mantendo os pixels mais claros
            cuda::max(gpu_frame, next_frame, gpu_frame);
        }

        // 17. Converter o frame para escala de cinzas
        cuda::cvtColor(gpu_frame, gpu_gray_frame, COLOR_BGR2GRAY);

        // 16. Normalizar a imagem dividindo cada valor de pixel pelo valor máximo da imagem e multiplicando pelo
        // valor máximo de intensidade possível:
        cuda::normalize(gpu_gray_frame, gpu_gray_frame, 0, 255, NORM_MINMAX, -1, noArray(), cuda::Stream::Null());

        // 18. Redimensionar a imagem para frame_sizexframe_size pixels

        // Define a largura e altura desejadas para a imagem redimensionada
        Size size(frame_size, frame_size);

        // Calcula a proporção entre a largura e altura da imagem original e da imagem desejada
        // A proporção é calculada como o mínimo entre a razão da largura original pela largura desejada
        // e a razão da altura original pela altura desejada
        float ratio = std::min((float)size.width / gpu_gray_frame.cols, (float)size.height / gpu_gray_frame.rows);

        // Calcula as novas dimensões para a imagem redimensionada, de forma que ela mantenha as proporções originais
        Size newSize(static_cast<int>(gpu_gray_frame.cols * ratio), static_cast<int>(gpu_gray_frame.rows * ratio));

        // Redimensiona a imagem em escala de cinza para as novas dimensões calculadas
        cuda::resize(gpu_gray_frame, gpu_gray_frame, newSize);

        Mat result;
        gpu_gray_frame.download(result);
        // 19. Salvar o quadro final do intervalo
        std::string filename = save_files_path + std::to_string((int)start_time) + "_" + std::to_string((int)end_time) + ".png";
        imwrite(filename, result);
    }

    // 20. Liberar recursos e encerrar o programa
    cap.release();
    destroyAllWindows();
    return 0;
}
