// alocati dinamic un vector vect de n elemente de tip struct particle
// verificati daca operatia a reusit
// cate elemente puteti aloca maxim (aproximativ)?
//   -> porniti de la 1.000.000 elemente; inmultiti cu 10

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>     // provides int8_t, uint8_t, int16_t etc.
#include <stdlib.h>

typedef struct {
    int8_t v_x, v_y, v_z;
} particle;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("apelati cu %s <n>\n", argv[0]);
        return -1;
    }

    long n = atol(argv[1]);

    particle *vect = malloc(sizeof(particle) * n);
    if (vect == NULL)
        return 12;

    // generate some particles
    for (long i = 1; i < n; i *= 2) {
        vect[i].v_x = (uint8_t) rand();
        vect[i].v_y = (uint8_t) rand();
        vect[i].v_z = (uint8_t) rand();
    }

    // compute max particle speed
    float max_speed = 0.0f;
    for (long i = 1; i < n; i *= 2) {
        float speed = sqrt(vect[i].v_x * vect[i].v_x +
                           vect[i].v_y * vect[i].v_y +
                           vect[i].v_z * vect[i].v_z);
        if (max_speed < speed) max_speed = speed;
    }

    // print result
    printf("viteza maxima este: %f\n", max_speed);

    free(vect);

    return 0;
}
