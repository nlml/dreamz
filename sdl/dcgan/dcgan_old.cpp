#include <torch/torch.h>
#include <iostream>
#include <GL/glut.h>
#include <GL/glut.h>
#include <iostream>
#include <time.h>
#include <chrono>
#include <thread>


#define WIDTH 100
#define HEIGHT 100

unsigned char texture[WIDTH][HEIGHT][3];


bool tensor2itk(torch::Tensor &t, int val)
{
    t = t.toType(torch::kUInt8);
    std::cout << "tensor dtype = " << t.dtype() << std::endl;
    // unsigned char *array = t.data<unsigned char, 2>();

    // assert t is 2-dimensional and holds unsigned chars.
    auto array = t.accessor<unsigned char, 2>();
    for(int i = 0; i < array.size(0); i++)
    {
        for(int j = 0; j < array.size(0); j++)
        {
            // use the accessor array to get tensor data.
            texture[i][j][0] = array[i][j] * val;
            // std::cout << static_cast<unsigned>( texture[i][j][0] ) << std::endl;
        }
	}
	return true;
}

void renderScene()
{
    clock_t tStart = clock();
    // render the texture here
    // const unsigned int r = rand() % 256;
    // const unsigned int g = rand() % 256;
    // const unsigned int b = rand() % 256;
    // for( unsigned int x = 0; x < WIDTH; x++ )
    // {
    //     for( unsigned int y = 0; y < HEIGHT; y++ )
    //     {
    //         texture[x][y][1] = g;
    //     	std::cout << static_cast<unsigned>(texture[x][y][1]) << std::endl;
    //         texture[x][y][0] = b;
    //         texture[x][y][2] = g;
    //     }
    // }

    glEnable (GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glTexImage2D (
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        WIDTH,
        HEIGHT,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        &texture[0][0][0]
    );

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(-1.0, -1.0);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f( 1.0, -1.0);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f( 1.0,  1.0);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(-1.0,  1.0);
    glEnd();

    glFlush();
    glutSwapBuffers();
    /* Do your stuff here */
    printf("Time taken: %.6fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
}



int main(int argc, char **argv)
{
    torch::Tensor tensor = torch::ones({100, 100});
    std::cout << tensor2itk(tensor, 200) << std::endl;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);

    glutInitWindowPosition(1000, 100);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow(" ");

    glutDisplayFunc(renderScene);
    std::cout << "ytooo" << std::endl;

    glutMainLoop();
	std::this_thread::sleep_for(std::chrono::milliseconds(1));
    std::cout << tensor2itk(tensor, 100) << std::endl;
    glutPostRedisplay();


    return 0;
}
