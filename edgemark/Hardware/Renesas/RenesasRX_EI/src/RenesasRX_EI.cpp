/***********************************************************************
*
*  FILE        : RenesasRX_EI.cpp
*  DATE        : 2024-08-07
*  DESCRIPTION : Main Program
*
*  NOTE:THIS IS A TYPICAL EXAMPLE.
*
***********************************************************************/
#ifdef CPPAPP
extern "C" {
#endif
#include "r_smc_entry.h"
#ifdef CPPAPP
}
#endif
#include "benchmark.h"

#ifdef CPPAPP
//Initialize global constructors
extern void __main()
{
  static int initialized;
  if (! initialized)
    {
      typedef void (*pfunc) ();
      extern pfunc __ctors[];
      extern pfunc __ctors_end[];
      pfunc *p;

      initialized = 1;
      for (p = __ctors_end; p > __ctors; )
    (*--p) ();

    }
}
#endif

int main(void) {

    benchmark(10, 0);

    while(1) {
    	;
    }
    return 0;
}
