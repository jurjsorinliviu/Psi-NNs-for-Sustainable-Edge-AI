/* main.c -- run the exported model on a Cortex-M core and report:
 *   - executed instructions (SysTick under QEMU -icount shift=0; see run_mcu_bench.py)
 *   - the model's actual fp32/int8 outputs, so numerical accuracy is measured
 *     on the target rather than assumed from the PyTorch model
 *
 * Identical source builds for Cortex-M4 (nRF52840 core) and Cortex-M7 (STM32H7 core).
 * The same firmware would run on real silicon; only the cycle source would change
 * (DWT->CYCCNT, which QEMU does not implement).
 */

#include <stdint.h>
#include "model.h"
#include "test_data.h"

#define UART0_DATA  (*(volatile uint32_t *)0x40004000)
#define UART0_STATE (*(volatile uint32_t *)0x40004004)
#define UART0_CTRL  (*(volatile uint32_t *)0x40004008)

#define SYST_CSR   (*(volatile uint32_t *)0xE000E010)
#define SYST_RVR   (*(volatile uint32_t *)0xE000E014)
#define SYST_CVR   (*(volatile uint32_t *)0xE000E018)
#define TICKS(a, b) (((a) - (b)) & 0x00FFFFFFu)

/* newlib's libm (tanhf) references __errno; supply the bare-metal stub */
int *__errno(void) { static int e; return &e; }

static void uart_init(void) { UART0_CTRL = 1u; }
static void uart_putc(char c) {
    while (UART0_STATE & 1u) {}
    UART0_DATA = (uint32_t)c;
}
static void uart_puts(const char *s) { while (*s) uart_putc(*s++); }

static void uart_i32(int32_t v) {
    char b[13]; int i = 12; b[i--] = 0;
    uint32_t u = (v < 0) ? (uint32_t)(-v) : (uint32_t)v;
    if (!u) b[i--] = '0';
    while (u) { b[i--] = (char)('0' + (u % 10u)); u /= 10u; }
    if (v < 0) b[i--] = '-';
    uart_puts(&b[i + 1]);
}

/* outputs are in [-1, 1]; print as fixed-point micro-units (6 decimals) so the host
   can recompute relative L2 and the antisymmetry residual from real device output */
static void uart_fixed(float f) {
    int32_t q = (int32_t)(f * 1000000.0f + (f >= 0 ? 0.5f : -0.5f));
    uart_i32(q);
}

static float outputs[NTEST];

int main(void) {
    uart_init();

    SYST_RVR = 0x00FFFFFFu;
    SYST_CVR = 0;
    SYST_CSR = 0x5u;                       /* enable, processor clock */

    /* warm the code path so the measurement excludes first-touch effects */
    float warm[MODEL_NOUT];
    MODEL_FORWARD(test_in[0], warm);

    uint32_t t0 = SYST_CVR;
    for (int k = 0; k < NTEST; k++) {
        float o[MODEL_NOUT];
        MODEL_FORWARD(test_in[k], o);
        outputs[k] = o[0];
    }
    uint32_t t1 = SYST_CVR;

    uart_puts("TICKS ");   uart_i32((int32_t)TICKS(t0, t1));
    uart_puts("\nNTEST ");  uart_i32(NTEST);
    uart_puts("\nOUT\n");
    for (int k = 0; k < NTEST; k++) { uart_fixed(outputs[k]); uart_puts("\n"); }
    uart_puts("DONE\n");
    for (;;) {}
}

/* ---- startup ---- */
extern uint32_t _estack;
void Reset_Handler(void) {
    extern uint32_t _sdata, _edata, _sidata, _sbss, _ebss;
    uint32_t *src = &_sidata, *dst = &_sdata;
    while (dst < &_edata) *dst++ = *src++;
    for (dst = &_sbss; dst < &_ebss;) *dst++ = 0;
    *(volatile uint32_t *)0xE000ED88 |= (0xFu << 20);   /* CP10/CP11: enable FPU */
    __asm__ volatile("dsb; isb");
    main();
    for (;;) {}
}
static void Default_Handler(void) { for (;;) {} }

__attribute__((section(".isr_vector"), used))
void (*const vector_table[])(void) = {
    (void (*)(void))&_estack,
    Reset_Handler,
    Default_Handler, Default_Handler, Default_Handler, Default_Handler,
};
