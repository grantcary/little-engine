#include <math.h>

struct shape_2D {
    float w, h;
};

struct point_2D {
    float x, y;
};

struct quadrilateral_2D {
    struct point_2D bl, tr, tl, br, origin;
};

struct quadrilateral_2D init_rectangle(int target_origin_x, float target_origin_y, float width, float height) {
    // left, right, top, bottom
    float l = target_origin_x - ((float) width/2);
    float r = target_origin_x + ((float) width/2);
    float t = target_origin_y + ((float) height/2);;
    float b = target_origin_y - ((float) height/2);;

    struct point_2D pnt_bl = {l, b};
    struct point_2D pnt_tr = {r, t};
    struct point_2D pnt_tl = {l, t};
    struct point_2D pnt_br = {r, b};

    struct point_2D origin = {target_origin_x, target_origin_y};
    struct quadrilateral_2D rectangle = {pnt_bl, pnt_tr, pnt_tl, pnt_br, origin};

    return rectangle;
}

struct quadrilateral_2D translate(struct quadrilateral_2D rectangle, float target_origin_x, float target_origin_y) {
    // calculate slope between origins
    float x_delta = target_origin_x - rectangle.origin.x;
    float y_delta = target_origin_y - rectangle.origin.y;

    // add deltas to respective coordinates
    struct point_2D point_bl = {rectangle.bl.x + x_delta, rectangle.bl.y + y_delta};
    struct point_2D point_tr = {rectangle.tr.x + x_delta, rectangle.tr.y + y_delta};
    struct point_2D point_tl = {rectangle.tl.x + x_delta, rectangle.tl.y + y_delta};
    struct point_2D point_br = {rectangle.br.x + x_delta, rectangle.br.y + y_delta};
    struct point_2D origin = {target_origin_x, target_origin_y};

    struct quadrilateral_2D target_origin = {point_bl, point_tr, point_tl, point_br, origin};

    return target_origin;
}

struct quadrilateral_2D rotate(struct quadrilateral_2D rect, float deg) {
    float x = rect.origin.x;
    float y = rect.origin.y;
    
    float rad = deg*(M_PI/180);
    float c = cos(rad);
    float s = sin(rad);

    struct point_2D point_bl = {x + ((rect.bl.x - x) * c) - ((rect.bl.y - y) * s), y + ((rect.bl.x - x) * s) + ((rect.bl.y - y) * c)};
    struct point_2D point_tr = {x + ((rect.tr.x - x) * c) - ((rect.tr.y - y) * s), y + ((rect.tr.x - x) * s) + ((rect.tr.y - y) * c)};
    struct point_2D point_tl = {x + ((rect.tl.x - x) * c) - ((rect.tl.y - y) * s), y + ((rect.tl.x - x) * s) + ((rect.tl.y - y) * c)};
    struct point_2D point_br = {x + ((rect.br.x - x) * c) - ((rect.br.y - y) * s), y + ((rect.br.x - x) * s) + ((rect.br.y - y) * c)};

    return (struct quadrilateral_2D) {point_bl, point_tr, point_tl, point_br, rect.origin};
}
