#include <GL/glut.h>
#include <GL/glut.h>
#include <iostream>

#define WIDTH 100
#define HEIGHT 100

unsigned char texture[WIDTH][HEIGHT][3];

void renderScene()
{
    // render the texture here
    const unsigned int r = rand() % 256;
    const unsigned int g = rand() % 256;
    const unsigned int b = rand() % 256;
    for( unsigned int x = 0; x < WIDTH; x++ )
    {
        for( unsigned int y = 0; y < HEIGHT; y++ )
        {
            texture[x][y][0] = b;
            texture[x][y][1] = g;
            texture[x][y][2] = g;
        }
    }

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
}

int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);

    glutInitWindowPosition(100, 100);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow(" ");

    glutDisplayFunc(renderScene);

    glutMainLoop();

    return 0;
}
