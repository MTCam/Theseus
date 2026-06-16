#pragma once
#include <chrono>
#include <cstdio>

namespace Theseus
{
  class ScopedTimer
  {
  public:
#ifdef ENABLE_TIMERS
    explicit ScopedTimer(const char *name)
      : name_(name),
	start_(clock::now()) {}

    ~ScopedTimer()
    {
      auto end = clock::now();
      double ms = std::chrono::duration<double, std::milli>(end - start_).count();
      std::printf("[TIMER] %s : %.6f ms\n", name_, ms);
    }
#else
    explicit ScopedTimer(const char *name)
      : name_(name),
	start_() {}
    ~ScopedTimer(){};
#endif
  private:
    using clock = std::chrono::steady_clock;
    const char *name_;
    clock::time_point start_;
  };

}
