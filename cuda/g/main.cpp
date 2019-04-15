#include <GL/glut.h>
#include <GLFW/glfw3.h>
#include <iostream>
using namespace std;

// void displayMe(void)
// {
//     glClear(GL_COLOR_BUFFER_BIT);
//     glBegin(GL_POLYGON);
//         glVertex3f(0.5, 0.0, 0.5);
//         glVertex3f(0.5, 0.0, 0.0);
//         glVertex3f(0.0, 0.5, 0.0);
//         glVertex3f(0.0, 0.0, 0.5);
//     glEnd();
//     glFlush();
// }

int main(int argc, char** argv)
{
    int count = 0;
    GLFWmonitor** monitors = glfwGetMonitors(&count);
    // glutInit(&argc, argv);
    // glutInitDisplayMode(GLUT_SINGLE);
    // glutInitWindowSize(400, 300);
    // glutInitWindowPosition(100, 100);
    // glutCreateWindow("Hello world!");
    // glutDisplayFunc(displayMe);
    // glutMainLoop();
    cout << "Hello, " << 2 << " is the half of " << count << "!\n";
    return 0;
}
