#include <stdio.h>
#include <math.h>

struct shape_3D {
    float w, h, d;
};

struct point_3D {
    float x, y, z;
};

struct sin_cos {
    float c, s;
};

struct cuboid {
    struct point_3D f_bl, f_tr, f_tl, f_br, b_bl, b_tr, b_tl, b_br, origin;
};

struct cuboid init_box(struct point_3D origin, struct shape_3D shape) {
    // distances from origin
    float left = origin.x - (shape.w/2);
    float right = origin.x + (shape.w/2);
    float top = origin.y + (shape.h/2);;
    float bottom = origin.y - (shape.h/2);
    float front = origin.z + (shape.d/2);
    float back = origin.z - (shape.d/2);

    // FRONT
    struct point_3D p_f_bl = {left, bottom, front};
    struct point_3D p_f_tr = {right, top, front};
    struct point_3D p_f_tl = {left, top, front};
    struct point_3D p_f_br = {right, bottom, front};

    // BACK
    struct point_3D p_b_bl = {left, bottom, back};
    struct point_3D p_b_tr = {right, top, back};
    struct point_3D p_b_tl = {left, top, back};
    struct point_3D p_b_br = {right, bottom, back};
    
    struct cuboid box = {p_f_bl, p_f_tr, p_f_tl, p_f_br, p_b_bl, p_b_tr, p_b_tl, p_b_br, origin};
    
    return box;
}

struct cuboid set_new_origin(struct cuboid box, struct point_3D origin) {
    // calculate slope between origins
    float x_delta = origin.x - box.origin.x;
    float y_delta = origin.y - box.origin.y;
    float z_delta = origin.z - box.origin.z;

    // FRONT
    struct point_3D p_f_bl = {box.f_bl.x + x_delta, box.f_bl.y + y_delta, box.f_bl.z + z_delta};
    struct point_3D p_f_tr = {box.f_tr.x + x_delta, box.f_tr.y + y_delta, box.f_bl.z + z_delta};
    struct point_3D p_f_tl = {box.f_tl.x + x_delta, box.f_tl.y + y_delta, box.f_bl.z + z_delta};
    struct point_3D p_f_br = {box.f_br.x + x_delta, box.f_br.y + y_delta, box.f_bl.z + z_delta};

    // BACK
    struct point_3D p_b_bl = {box.b_bl.x + x_delta, box.b_bl.y + y_delta, box.b_bl.z + z_delta};
    struct point_3D p_b_tr = {box.b_tr.x + x_delta, box.b_tr.y + y_delta, box.b_bl.z + z_delta};
    struct point_3D p_b_tl = {box.b_tl.x + x_delta, box.b_tl.y + y_delta, box.b_bl.z + z_delta};
    struct point_3D p_b_br = {box.b_br.x + x_delta, box.b_br.y + y_delta, box.b_bl.z + z_delta};

    box = (struct cuboid) {p_f_bl, p_f_tr, p_f_tl, p_f_br, p_b_bl, p_b_tr, p_b_tl, p_b_br, origin};

    return box;
}

struct cuboid rotate_2D(struct cuboid box, char axis, int degrees) {
    float x = box.origin.x;
    float y = box.origin.y;
    float z = box.origin.z;

    float rad = degrees*(M_PI/180);
    float c = cos(rad);
    float s = sin(rad);

    switch (axis) {
        case 'x': {
            // default rotation direction: counter clock-wise
            struct point_3D p_f_bl = {box.f_bl.x, y + ((box.f_bl.y - y) * c) - ((box.f_bl.z - z) * s), z + ((box.f_bl.y - y) * s) + ((box.f_bl.z - z) * c)};
            struct point_3D p_f_tr = {box.f_tr.x, y + ((box.f_tr.y - y) * c) - ((box.f_tr.z - z) * s), z + ((box.f_tr.y - y) * s) + ((box.f_tr.z - z) * c)};
            struct point_3D p_f_tl = {box.f_tl.x, y + ((box.f_tl.y - y) * c) - ((box.f_tl.z - z) * s), z + ((box.f_tl.y - y) * s) + ((box.f_tl.z - z) * c)};
            struct point_3D p_f_br = {box.f_br.x, y + ((box.f_br.y - y) * c) - ((box.f_br.z - z) * s), z + ((box.f_br.y - y) * s) + ((box.f_br.z - z) * c)};

            struct point_3D p_b_bl = {box.b_bl.x, y + ((box.b_bl.y - y) * c) - ((box.b_bl.z - z) * s), z + ((box.b_bl.y - y) * s) + ((box.b_bl.z - z) * c)};
            struct point_3D p_b_tr = {box.b_tr.x, y + ((box.b_tr.y - y) * c) - ((box.b_tr.z - z) * s), z + ((box.b_tr.y - y) * s) + ((box.b_tr.z - z) * c)};
            struct point_3D p_b_tl = {box.b_tl.x, y + ((box.b_tl.y - y) * c) - ((box.b_tl.z - z) * s), z + ((box.b_tl.y - y) * s) + ((box.b_tl.z - z) * c)};
            struct point_3D p_b_br = {box.b_br.x, y + ((box.b_br.y - y) * c) - ((box.b_br.z - z) * s), z + ((box.b_br.y - y) * s) + ((box.b_br.z - z) * c)};

            return (struct cuboid) {p_f_bl, p_f_tr, p_f_tl, p_f_br, p_b_bl, p_b_tr, p_b_tl, p_b_br, box.origin};
        }
        case 'y': {
            // default rotation direction: clock-wise
            struct point_3D p_f_bl = {x + ((box.f_bl.x - x) * c) - ((box.f_bl.z - z) * s), box.f_bl.y, z + ((box.f_bl.x - x) * s) + ((box.f_bl.z - z) * c)};
            struct point_3D p_f_tr = {x + ((box.f_tr.x - x) * c) - ((box.f_tr.z - z) * s), box.f_tr.y, z + ((box.f_tr.x - x) * s) + ((box.f_tr.z - z) * c)};
            struct point_3D p_f_tl = {x + ((box.f_tl.x - x) * c) - ((box.f_tl.z - z) * s), box.f_tl.y, z + ((box.f_tl.x - x) * s) + ((box.f_tl.z - z) * c)};
            struct point_3D p_f_br = {x + ((box.f_br.x - x) * c) - ((box.f_br.z - z) * s), box.f_br.y, z + ((box.f_br.x - x) * s) + ((box.f_br.z - z) * c)};

            struct point_3D p_b_bl = {x + ((box.b_bl.x - x) * c) - ((box.b_bl.z - z) * s), box.b_bl.y, z + ((box.b_bl.x - x) * s) + ((box.b_bl.z - z) * c)};
            struct point_3D p_b_tr = {x + ((box.b_tr.x - x) * c) - ((box.b_tr.z - z) * s), box.b_tr.y, z + ((box.b_tr.x - x) * s) + ((box.b_tr.z - z) * c)};
            struct point_3D p_b_tl = {x + ((box.b_tl.x - x) * c) - ((box.b_tl.z - z) * s), box.b_tl.y, z + ((box.b_tl.x - x) * s) + ((box.b_tl.z - z) * c)};
            struct point_3D p_b_br = {x + ((box.b_br.x - x) * c) - ((box.b_br.z - z) * s), box.b_br.y, z + ((box.b_br.x - x) * s) + ((box.b_br.z - z) * c)};

            return (struct cuboid) {p_f_bl, p_f_tr, p_f_tl, p_f_br, p_b_bl, p_b_tr, p_b_tl, p_b_br, box.origin};
        }
        case 'z': {
            // default rotation direction: counter clock-wise
            struct point_3D p_f_bl = {x + ((box.f_bl.x - x) * c) - ((box.f_bl.y - y) * s), y + ((box.f_bl.x - x) * s) + ((box.f_bl.y - y) * c), box.f_bl.z};
            struct point_3D p_f_tr = {x + ((box.f_tr.x - x) * c) - ((box.f_tr.y - y) * s), y + ((box.f_tr.x - x) * s) + ((box.f_tr.y - y) * c), box.f_tr.z};
            struct point_3D p_f_tl = {x + ((box.f_tl.x - x) * c) - ((box.f_tl.y - y) * s), y + ((box.f_tl.x - x) * s) + ((box.f_tl.y - y) * c), box.f_tl.z};
            struct point_3D p_f_br = {x + ((box.f_br.x - x) * c) - ((box.f_br.y - y) * s), y + ((box.f_br.x - x) * s) + ((box.f_br.y - y) * c), box.f_br.z};

            struct point_3D p_b_bl = {x + ((box.b_bl.x - x) * c) - ((box.b_bl.y - y) * s), y + ((box.b_bl.x - x) * s) + ((box.b_bl.y - y) * c), box.b_bl.z};
            struct point_3D p_b_tr = {x + ((box.b_tr.x - x) * c) - ((box.b_tr.y - y) * s), y + ((box.b_tr.x - x) * s) + ((box.b_tr.y - y) * c), box.b_tr.z};
            struct point_3D p_b_tl = {x + ((box.b_tl.x - x) * c) - ((box.b_tl.y - y) * s), y + ((box.b_tl.x - x) * s) + ((box.b_tl.y - y) * c), box.b_tl.z};
            struct point_3D p_b_br = {x + ((box.b_br.x - x) * c) - ((box.b_br.y - y) * s), y + ((box.b_br.x - x) * s) + ((box.b_br.y - y) * c), box.b_br.z};

            return (struct cuboid) {p_f_bl, p_f_tr, p_f_tl, p_f_br, p_b_bl, p_b_tr, p_b_tl, p_b_br, box.origin};
        }
        default: {
            return box;
        }
    }
}

void print_point(struct point_3D vertex) {
    printf("(%f, %f, %f)", vertex.x, vertex.y, vertex.z);
}

void print_rectangle(struct point_3D tl, struct point_3D tr, struct point_3D bl, struct point_3D br) {
    printf("tl: ");
    print_point(tl);
    printf("    tr: ");
    print_point(tr);
    printf("\nbl: ");
    print_point(bl);
    printf("    br: ");
    print_point(br);
}

void print_shape(struct cuboid box) {
    printf("\nFront:    x         y         z                  x         y         z\n");
    print_rectangle(box.f_tl, box.f_tr, box.f_bl, box.f_br);

    printf("\nBack:\n");
    print_rectangle(box.b_tl, box.b_tr, box.b_bl, box.b_br);;

    printf("\nOrigin: \n");
    print_point(box.origin);
    printf("\n");
}

struct point_3D rotate_3D(float a_rad, float b_rad, float g_rad) {
    a_rad = a_rad*(M_PI/180);
    struct sin_cos a = {cos(a_rad), sin(a_rad)};

    b_rad = b_rad*(M_PI/180);
    struct sin_cos b = {cos(b_rad), sin(b_rad)};

    g_rad = g_rad*(M_PI/180);
    struct sin_cos g = {cos(g_rad), sin(g_rad)};

    struct point_3D p = {0.000000, 0.000000, 2.000000};
    float x, y, z;
    x = p.x * (b.c * g.c) + p.y * ((a.s * b.s * g.c) - (a.c * g.s)) + p.z * ((a.c * b.s * g.c) + (a.s * g.s));
    y = p.x * (b.c * g.s) + p.y * ((a.s * b.s * g.s) + (a.c * g.c)) + p.z * ((a.c * b.s * g.s) - (a.s * g.c));
    z = p.x * -b.s + p.y * (a.s * b.c) + p.z * (a.c * b.c);

    return (struct point_3D) {x, y, z};
}

int main() {
    float x, y, z, d;
    char a;

    // printf("Origin: ");
    // scanf("%f %f %f", &x, &y, &z);

    // printf("Axis: ");
    // scanf(" %c", &a);

    // printf("Degrees: ");
    // scanf("%f", &d);
    
    struct point_3D origin = {0, 0, 0};
    struct shape_3D shape = {4, 4, 4};
    struct cuboid cube = init_box(origin, shape);

    cube = (struct cuboid) rotate_2D(cube, 'x', 45);
    cube = (struct cuboid) rotate_2D(cube, 'y', 90);

    print_shape(cube);

    print_point(rotate_3D(45, 0, 0));

    // origin = (struct point_3D) {4, 5, 2};
    // cube = (struct cuboid) set_new_origin(cube, origin);

    // print_shape(cube);

    return 0;
}