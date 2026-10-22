/* M0a: proves the qwen36 CUDA Segment engine object links into the segment
 * runtime with the CUDA backend enabled. Unlike the CPU registration test this
 * builds only qwen36 (the engine that will carry the CUDA dense-chain adapter);
 * the other CPU adapters stay out so this gate is meaningful on CUDA-only hosts
 * without pulling in the whole CPU segment object set. */
#include "../segment_adapters.h"
#include "../segment_runtime.h"

#include <assert.h>
#include <stdio.h>

int main(void) {
    assert(coli_qwen36_segment_adapter_register() == 0);
    assert(coli_segment_adapter_count() == 1);
    assert(coli_segment_adapter_lookup("qwen36") != NULL);
    puts("qwen36 CUDA Segment adapter registers: ok");
    return 0;
}
