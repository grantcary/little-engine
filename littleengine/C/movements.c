#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include "primatives_2D.h"

void print_shape(struct quadrilateral_2D r) {
    printf("tl(%f, %f)  tr(%f, %f)\n", r.tl.x, r.tl.y, r.tr.x, r.tr.y);
    printf("bl(%f, %f)  br(%f, %f)\n", r.bl.x, r.bl.y, r.br.x, r.br.y);
}

void print_origin(int d, float x, float y) {
    printf("d: %d, x: %f, y: %f\n", d, x, y);
}

void move_origin_along_radius(struct quadrilateral_2D r, int scale) {
    for (int deg = 0; deg < 360; deg++) {
        float rad = deg*(M_PI/180);
        float x = cos(rad)*scale;
        float y = sin(rad)*scale;

        r = translate(r, x, y);
        print_shape(r);
        // print_origin(deg, x, y);
        sleep(0.5); // doesnt work right now
    }
}

int main() {
    float degrees;
    float x, y;

    printf("Origin: ");
    scanf("%f %f", &x, &y);

    printf("Degrees: ");
    scanf("%f", &degrees);

    struct quadrilateral_2D r = init_rectangle(x, y, 4, 4);
    r = (struct quadrilateral_2D) rotate(r, degrees);
    print_shape(r);

    return 0;
}