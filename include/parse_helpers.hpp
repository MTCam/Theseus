#include <algorithm>
#include <cctype>
#include <string>

auto to_lower = [](std::string s)
 {
   std::transform(s.begin(), s.end(), s.begin(),
		  [](unsigned char c) {
		    return static_cast<char>(std::tolower(c));
		  });
   return s;
 };

auto starts_with = [](const std::string &s, const std::string &prefix)
 {
   return s.rfind(prefix, 0) == 0;
 };
