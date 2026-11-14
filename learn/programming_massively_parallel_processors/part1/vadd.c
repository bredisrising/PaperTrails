#include <stdlib.h>

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    for (int i = 0; i < n; i++) {
        C_h[i] = A_h[i] + B_h[i];
    }
}


int main() {
    int n = 1 * 1000000;

    float* A_h = (float*) malloc(n * sizeof(float));
    float* B_h = (float*) malloc(n * sizeof(float));

    // init
    for (int i = 0; i < n; i++) {
        A_h[i] = 1; 
        A_h[i] = 2; 
    }
    

    float* C_h = (float*) malloc(n * sizeof(float));

    vecAdd(A_h, B_h, C_h, n);

    
    free(A_h);
    free(B_h);
    free(C_h);
}
