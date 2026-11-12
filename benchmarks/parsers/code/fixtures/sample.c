#include <stdio.h>
#include <stdlib.h>

int multiply(int a, int b) {
    return a * b;
}

int main(int argc, char** argv) {
    printf("Result: %d\n", multiply(6, 7));
    return 0;
}