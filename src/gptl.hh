#pragma once

#ifdef GPTL
#include "gptl.h"
#include "gptlmpi.h"
#else
#define GPTLinitialize() do {} while(0)
#define GPTLfinalize() do {} while(0)
#define GPTLpr_summary(...) do {} while(0)
#define GPTLstart(...) do {} while(0)
#define GPTLstop(...) do {} while(0)
#endif
