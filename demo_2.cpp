//normal template code for placing 3d objects

#include <windows.h>
#include <GL/glut.h>
#include <stdlib.h>
#include <iostream>

using namespace std;
void init()
{
    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-4.0, 4.0, -4.0, 4.0, 1.0, 10.0);
    glOrtho(-4.0, 4.0, -4.0,  4.0, 1.0, 10.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity() ;
    gluLookAt(0., 0., 5., 0., 0., 0., 0., 1., 0.);
}
static int slices = 16;
static int stacks = 16;

/* GLUT callback Handlers */

static void resize(int width, int height)
{
    const float ar = (float) width / (float) height;

    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    if (ar < 1.0)
    {
        glOrtho(-4., 4., -4./ar, 4./ar, 1., 100.);
    }
    else
    {
        glOrtho(-4.*ar, 4.*ar, -4., 4., 1., 100.);
    }
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity() ;
    gluLookAt(0., 0., 5., 0., 0., 0., 0., 1., 0.);

}

static void display(void)
{
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glutWireSphere(2,100,100);

    glutSwapBuffers();
}


static void key(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 27 :
        case 'q':
            exit(0);
            break;

        case '+':
            slices++;
            stacks++;
            break;

        case '-':
            if (slices>3 && stacks>3)
            {
                slices--;
                stacks--;
            }
            break;
    }

    glutPostRedisplay();
}

static void idle(void)
{
    glutPostRedisplay();
}

/* Program entry point */

int main(int argc, char *argv[])
{
    glutInit(&argc, argv);
    glutInitWindowSize(640,480);
    glutInitWindowPosition(10,10);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

    glutCreateWindow("GLUT Shapes");

    glutReshapeFunc(resize);
    glutDisplayFunc(display);
    glutKeyboardFunc(key);

    init();

    glutMainLoop();

    return EXIT_SUCCESS;
}
