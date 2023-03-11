#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main(int argc, char **argv)
{
    // 1. Verificar se o número correto de argumentos foi passado
    if (argc != 6)
    {
        std::cout << "Use: ./normalize start_interval duration_in_seconds frame_size video_file_path.mp4 save_files_path/" << std::endl;
        return -1;
    }

    // 2. Extrair os argumentos de linha de comando
    int start_interval = std::stoi(argv[1]);
    int duration_in_seconds = std::stoi(argv[2]);
    int frame_size = std::stoi(argv[3]);
    std::string video_file_path = argv[4];
    std::string save_files_path = argv[5];

    // 3. Carregar o arquivo de vídeo
    cv::VideoCapture cap(video_file_path);

    // 4. Verificar se o arquivo foi carregado corretamente
    if (!cap.isOpened())
    {
        std::cout << "Erro ao abrir arquivo de vídeo" << std::endl;
        return -1;
    }

    // 5. Definir a taxa de quadros do vídeo
    double fps = cap.get(cv::CAP_PROP_FPS);

    // 6. Definir o número total de quadros no vídeo
    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);

    // 7. Definir o número de intervalos que serão criados
    int num_intervals = total_frames / (fps * duration_in_seconds);

    // 8. Loop para processar cada intervalo
    for (int i = start_interval; i < num_intervals; i++)
    {

        // 9. Definir o tempo de início e fim do intervalo
        double start_time = i * duration_in_seconds;
        double end_time = (i + 1) * duration_in_seconds;

        // 10. Posicionar o vídeo no tempo de início do intervalo
        cap.set(cv::CAP_PROP_POS_MSEC, start_time * 1000);

        // 11. Capturar o primeiro quadro do intervalo
        cv::Mat frame;
        cap.read(frame);

        // 12. Loop para processar cada quadro do intervalo
        for (int j = 1; j < (int)(fps * duration_in_seconds); j++)
        {

            // 13. Posicionar o vídeo no tempo do próximo quadro
            cap.set(cv::CAP_PROP_POS_MSEC, (start_time + j / fps) * 1000);

            // 14. Capturar o próximo quadro
            cv::Mat next_frame;
            cap.read(next_frame);

            // 15. Sobrepor os quadros mantendo os pixels mais claros
            for (int y = 0; y < frame.rows; y++)
            {
                for (int x = 0; x < frame.cols; x++)
                {
                    cv::Vec3b pixel1 = frame.at<cv::Vec3b>(y, x);
                    cv::Vec3b pixel2 = next_frame.at<cv::Vec3b>(y, x);
                    cv::Vec3b result_pixel;
                    for (int i = 0; i < 3; i++)
                    {
                        result_pixel[i] = std::max(pixel1[i], pixel2[i]);
                    }
                    frame.at<cv::Vec3b>(y, x) = result_pixel;
                }
            }
        }

        // 16. Normalizar a imagem dividindo cada valor de pixel pelo valor máximo da imagem e multiplicando pelo
        // valor máximo de intensidade possível:
        cv::normalize(frame, frame, 0, 255, cv::NORM_MINMAX);

        // 17. Converter o frame para escala de cinzas
        cv::Mat gray_frame;
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

        // 18. Redimensionar a imagem para frame_sizexframe_size pixels

        // Define a largura e altura desejadas para a imagem redimensionada
        cv::Size size(frame_size, frame_size);

        // Obtém as dimensões da imagem em escala de cinza
        cv::Size imageSize = gray_frame.size();

        // Calcula a proporção entre a largura e altura da imagem original e da imagem desejada
        // A proporção é calculada como o mínimo entre a razão da largura original pela largura desejada
        // e a razão da altura original pela altura desejada
        float ratio = std::min((float)size.width / imageSize.width, (float)size.height / imageSize.height);

        // Calcula as novas dimensões para a imagem redimensionada, de forma que ela mantenha as proporções originais
        cv::Size newSize(imageSize.width * ratio, imageSize.height * ratio);

        // Redimensiona a imagem em escala de cinza para as novas dimensões calculadas
        cv::resize(gray_frame, gray_frame, newSize);

        // 19. Salvar o quadro final do intervalo
        std::string filename = save_files_path + std::to_string((int)start_time) + "_" + std::to_string((int)end_time) + ".png";
        cv::imwrite(filename, gray_frame);
    }

    // 20. Liberar recursos e encerrar o programa
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
