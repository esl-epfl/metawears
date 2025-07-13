#include "mobilenet.h"
#include "inference.h"

int main() {
    mobilenet net;
    setup_mobilenet(&net);
    mobilenet_inference(&net);
}

