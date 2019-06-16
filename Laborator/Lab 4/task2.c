#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>     // provides int8_t, uint8_t, int16_t etc.
#include <stdlib.h>
#include <assert.h>

typedef struct {
    int8_t v_x, v_y, v_z;
} particle;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("apelati cu %s <n>\n", argv[0]);
        return -1;
    }

    long n = atol(argv[1]);

    // alocati dinamic o matrice de n x n elemente de tip struct particle
    // verificati daca operatia a reusit
    particle *particles = malloc(n * n * sizeof(particle));
    assert(particles != NULL);

    // populati matricea alocata astfel:
    // *liniile pare contin particule cu toate componentele vitezei pozitive
    //   -> folositi modulo 128 pentru a limita rezultatului lui rand()
    // *liniile impare contin particule cu toate componentele vitezi negative
    //   -> folositi modulo 129 pentru a limita rezultatului lui rand()
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            if (i % 2 == 0) {
                particles[i * n + j] = (particle) {(int8_t) (rand() % 128),
                                                   (int8_t) (rand() % 128),
                                                   (int8_t) (rand() % 128)};
            } else {
                particles[i * n + j] = (particle) {(int8_t) (-(rand() % 129)),
                                                   (int8_t) (-(rand() % 129)),
                                                   (int8_t) (-(rand() % 129))};
            }
        }
    }

    int8_t *t = (int8_t *)particles;
    // scalati vitezele tuturor particulelor cu 0.5
    //   -> folositi un cast la int8_t* pentru a parcurge vitezele fara
    //      a fi nevoie sa accesati individual componentele v_x, v_y, si v_z
    for (long i = 0; i < 3 * n * n; i++){
        t[i] =  (t[i] / 2);
    }

    // compute max particle speed
    float max_speed = 0.0f;
    for (long i = 0; i < n * n; ++i) {
        float speed = sqrt(particles[i].v_x * particles[i].v_x +
                           particles[i].v_y * particles[i].v_y +
                           particles[i].v_z * particles[i].v_z);
        if (max_speed < speed) max_speed = speed;
    }

    // print result
    printf("viteza maxima este: %f\n", max_speed);

    free(particles);

    return 0;
}

